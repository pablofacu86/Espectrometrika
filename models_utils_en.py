"""
models_utils.py
Motor de modelado supervisado (clasificación y regresión) y selección de
variables, para complementar chemo_utils.py (que cubre carga, pretratamiento
y análisis exploratorio).
"""

import datetime
import platform
import secrets
import optuna
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.experimental import enable_halving_search_cv  # noqa: F401 (enables the classes below)
from sklearn.model_selection import (
    StratifiedKFold, KFold, LeaveOneOut, cross_val_predict, cross_val_score, GridSearchCV, RandomizedSearchCV,
    HalvingGridSearchCV, HalvingRandomSearchCV, learning_curve,
    train_test_split,
)
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score,
    cohen_kappa_score, precision_score, recall_score, classification_report,
    roc_auc_score, matthews_corrcoef,
)
from xgboost import XGBClassifier, XGBRegressor


# -----------------------------------------------------------------------------
# CPU budget. Cloud containers often report the CPU count of the whole host (e.g. 32+)
# while the app is only allowed to use 1-2 CPUs. With n_jobs=-1 every parallel step then
# starts dozens of workers that fight for those few CPUs (and copy the data in memory):
# the app becomes SLOWER and can freeze or run out of memory. We detect the CPUs the
# container may really use and use at most that many (max 4) for the outer parallel loop;
# the models inside it stay single-threaded so the work is never multiplied.
# -----------------------------------------------------------------------------
import math
import os


def _cpus_efectivos():
    try:
        n = len(os.sched_getaffinity(0))
    except Exception:
        n = os.cpu_count() or 1
    try:                                    # cgroup v2
        cuota, periodo = open("/sys/fs/cgroup/cpu.max").read().split()[:2]
        if cuota != "max":
            n = min(n, max(1, math.ceil(int(cuota) / int(periodo))))
    except Exception:
        try:                                # cgroup v1
            cuota = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
            periodo = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
            if cuota > 0:
                n = min(n, max(1, math.ceil(cuota / periodo)))
        except Exception:
            pass
    return max(1, n)


N_JOBS = max(1, min(4, _cpus_efectivos()))
try:                                        # BLAS / OpenMP threads: same budget
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=N_JOBS)
except Exception:
    pass

optuna.logging.set_verbosity(optuna.logging.WARNING)  # keep Optuna's trial-by-trial log quiet


# =============================================================================
# WRAPPERS: PLS-DA y PCR (no existen "de fábrica" en scikit-learn con esta
# interfaz, así que se arman como wrappers compatibles con scikit-learn)
# =============================================================================

class MLPClassifierWrapper(BaseEstimator, ClassifierMixin):
    """
    Wrapper around MLPClassifier that encodes classes to integers
    internally and returns predictions using the original labels. This
    works around a real scikit-learn limitation: MLPClassifier with
    early_stopping=True crashes (TypeError inside its internal validation
    scoring) when given string class labels directly — encoding to
    integers first avoids that entirely, while still letting us use
    early_stopping (which is what keeps training from grinding on for the
    full max_iter budget on every fold when the network isn't improving —
    the main reason a Neural Network model can otherwise take several
    minutes per fold on limited hardware).
    """

    def __init__(self, hidden_layer_sizes=(50,), max_iter=500, early_stopping=True,
                 n_iter_no_change=15, alpha=0.0001, random_state=0):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.alpha = alpha
        self.random_state = random_state

    def fit(self, X, y):
        self.encoder_ = LabelEncoder()
        y_enc = self.encoder_.fit_transform(y)
        try:
            self.modelo_ = MLPClassifier(
                hidden_layer_sizes=self.hidden_layer_sizes, max_iter=self.max_iter,
                early_stopping=self.early_stopping, n_iter_no_change=self.n_iter_no_change,
                alpha=self.alpha, random_state=self.random_state,
            )
            self.modelo_.fit(X, y_enc)
        except ValueError as e:
            # early_stopping carves out an internal validation split (10% of
            # the training data by default) — with very few samples (a small
            # CV fold, a small dataset, or a rare class), that split can end
            # up too small or missing a class entirely, and sklearn raises
            # here instead of just proceeding. Retry once without
            # early_stopping in that specific case (max_iter is still capped,
            # so this doesn't reintroduce the "grinds forever" problem).
            if "validation set is too small" in str(e) or "least populated class" in str(e):
                self.modelo_ = MLPClassifier(
                    hidden_layer_sizes=self.hidden_layer_sizes, max_iter=self.max_iter,
                    early_stopping=False, random_state=self.random_state,
                )
                self.modelo_.fit(X, y_enc)
            else:
                raise
        self.classes_ = self.encoder_.classes_
        return self

    def predict(self, X):
        pred_enc = self.modelo_.predict(X)
        return self.encoder_.inverse_transform(pred_enc)

    def predict_proba(self, X):
        return self.modelo_.predict_proba(X)


class MLPRegressorWrapper(BaseEstimator, RegressorMixin):
    """
    Thin wrapper around MLPRegressor that retries without early_stopping if
    the internal validation split ends up too small for a given fold/dataset
    size (same safety fallback as MLPClassifierWrapper — see its docstring).
    """

    def __init__(self, hidden_layer_sizes=(50,), max_iter=500, early_stopping=True,
                 n_iter_no_change=15, alpha=0.0001, random_state=0):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.alpha = alpha
        self.random_state = random_state

    def fit(self, X, y):
        try:
            self.modelo_ = MLPRegressor(
                hidden_layer_sizes=self.hidden_layer_sizes, max_iter=self.max_iter,
                early_stopping=self.early_stopping, n_iter_no_change=self.n_iter_no_change,
                alpha=self.alpha, random_state=self.random_state,
            )
            self.modelo_.fit(X, y)
        except ValueError as e:
            if "validation set is too small" in str(e):
                self.modelo_ = MLPRegressor(
                    hidden_layer_sizes=self.hidden_layer_sizes, max_iter=self.max_iter,
                    early_stopping=False, random_state=self.random_state,
                )
                self.modelo_.fit(X, y)
            else:
                raise
        return self

    def predict(self, X):
        return self.modelo_.predict(X)


class PLSDAClassifier(BaseEstimator, ClassifierMixin):
    """
    PLS-DA: PLS de regresión sobre las clases codificadas como variables
    dummy (una columna por clase), clasificando cada muestra según la
    columna con mayor valor predicho.
    """

    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X, y):
        self.encoder_ = LabelBinarizer()
        Y_dummy = self.encoder_.fit_transform(y)
        # Con solo 2 clases, LabelBinarizer da una sola columna: la
        # completamos a 2 columnas para poder usar el mismo criterio de argmax.
        if Y_dummy.shape[1] == 1:
            Y_dummy = np.hstack([1 - Y_dummy, Y_dummy])
        self.classes_ = self.encoder_.classes_
        n_comp = min(self.n_components, X.shape[0] - 1, X.shape[1])
        self.pls_ = PLSRegression(n_components=max(1, n_comp))
        self.pls_.fit(X, Y_dummy)
        return self

    def decision_function(self, X):
        return self.pls_.predict(X)

    def predict_proba(self, X):
        """Pseudo-probabilidades vía softmax sobre la salida del PLS (no están
        calibradas, pero sirven para rankear y calcular AUC)."""
        raw = self.decision_function(X)
        raw = raw - raw.max(axis=1, keepdims=True)
        exp = np.exp(raw)
        return exp / exp.sum(axis=1, keepdims=True)

    def predict(self, X):
        pred = self.decision_function(X)
        idx = np.argmax(pred, axis=1)
        return self.classes_[idx]


class PCRRegressor(BaseEstimator, RegressorMixin):
    """Regresión de Componentes Principales: PCA seguido de regresión lineal."""

    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X, y):
        n_comp = min(self.n_components, X.shape[0] - 1, X.shape[1])
        self.pca_ = PCA(n_components=max(1, n_comp))
        scores = self.pca_.fit_transform(X)
        self.reg_ = LinearRegression()
        self.reg_.fit(scores, y)
        return self

    def predict(self, X):
        scores = self.pca_.transform(X)
        return self.reg_.predict(scores)


class XGBClassifierWrapper(BaseEstimator, ClassifierMixin):
    """
    Wrapper around XGBClassifier that encodes classes to integers
    internally (XGBoost requires this) and returns predictions using the
    original labels, so it behaves like every other classifier. Exposes
    its hyperparameters as REAL named __init__ parameters (not a generic
    **kwargs blob) — this is required for scikit-learn's parameter
    introspection (get_params/set_params/clone) to work correctly. Without
    it, every cross-validation fold silently re-clones the estimator with
    DEFAULT XGBoost settings instead of the ones actually configured here,
    since a generic **kwargs constructor isn't visible to get_params().
    """

    def __init__(self, n_estimators=300, max_depth=3, learning_rate=0.1,
                 eval_metric="mlogloss", random_state=0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.eval_metric = eval_metric
        self.random_state = random_state

    def fit(self, X, y):
        self.encoder_ = LabelEncoder()
        y_enc = self.encoder_.fit_transform(y)
        self.modelo_ = XGBClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            learning_rate=self.learning_rate, eval_metric=self.eval_metric,
            random_state=self.random_state, n_jobs=1,
            colsample_bytree=0.3, max_bin=64,   # fast on spectra with hundreds of variables
        )
        self.modelo_.fit(X, y_enc)
        self.classes_ = self.encoder_.classes_
        return self

    def predict(self, X):
        pred_enc = self.modelo_.predict(X)
        return self.encoder_.inverse_transform(pred_enc)

    def predict_proba(self, X):
        return self.modelo_.predict_proba(X)


# =============================================================================
# CATÁLOGO DE MODELOS
# =============================================================================

def crear_clasificadores(n_componentes_pls=5, random_state=0):
    """Devuelve un diccionario {nombre: instancia sin entrenar} de clasificadores."""
    return {
        "LDA": LinearDiscriminantAnalysis(),
        "PLS-DA": PLSDAClassifier(n_components=n_componentes_pls),
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=150, random_state=random_state, n_jobs=1),
        "SVM": SVC(kernel="rbf", probability=True, random_state=random_state),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "XGBoost": XGBClassifierWrapper(
            n_estimators=200, eval_metric="mlogloss", random_state=random_state,
        ),
        "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=1),
        "Neural Network (MLP)": MLPClassifierWrapper(
            hidden_layer_sizes=(50,), max_iter=500, early_stopping=True,
            n_iter_no_change=15, random_state=random_state,
        ),
    }


def crear_regresores(n_componentes_pls=5, random_state=0):
    """Devuelve un diccionario {nombre: instancia sin entrenar} de regresores."""
    return {
        "PLS": PLSRegression(n_components=n_componentes_pls),
        "PCR": PCRRegressor(n_components=n_componentes_pls),
        "Linear Regression (MLR)": LinearRegression(),
        "Lasso": Lasso(alpha=0.01, max_iter=20000),
        "Ridge": Ridge(alpha=1.0),
        "Elastic Net": ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=20000),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_features=0.2, random_state=random_state, n_jobs=1),
        "SVM (SVR)": SVR(kernel="rbf"),
        "Decision Tree": DecisionTreeRegressor(random_state=random_state),
        "XGBoost": XGBRegressor(n_estimators=200, colsample_bytree=0.3, max_bin=64, random_state=random_state, n_jobs=1),
        "KNN": KNeighborsRegressor(n_neighbors=5, n_jobs=1),
        "Neural Network (MLP)": MLPRegressorWrapper(
            hidden_layer_sizes=(50,), max_iter=500, early_stopping=True,
            n_iter_no_change=15, random_state=random_state,
        ),
    }


# =============================================================================
# GRILLAS DE HIPERPARÁMETROS Y OPTIMIZACIÓN
# =============================================================================

GRILLAS_CLASIFICACION = {
    "LDA": {"solver": ["svd", "lsqr"]},
    "PLS-DA": {"n_components": [2, 3, 5, 7, 10]},
    "Logistic Regression": {"C": [0.01, 0.1, 1, 10, 100]},
    "Naive Bayes": {"var_smoothing": [1e-9, 1e-7, 1e-5]},
    "Random Forest": {"n_estimators": [100, 200, 300], "max_depth": [None, 10, 20],
                       "max_features": ["sqrt", "log2", 0.3]},
    "SVM": {"C": [0.1, 1, 10, 100], "gamma": ["scale", "auto", 0.01, 0.1]},
    "Decision Tree": {"max_depth": [None, 3, 5, 10, 20]},
    "XGBoost": {"n_estimators": [100, 200, 300], "max_depth": [3, 5, 7], "learning_rate": [0.05, 0.1, 0.3]},
    "KNN": {"n_neighbors": [3, 5, 7, 9, 11]},
    "Neural Network (MLP)": {"hidden_layer_sizes": [(50,), (100,), (50, 50)], "alpha": [0.0001, 0.001, 0.01]},
}

GRILLAS_REGRESION = {
    "PLS": {"n_components": [2, 3, 5, 7, 10]},
    "PCR": {"n_components": [2, 3, 5, 7, 10]},
    "Lasso": {"alpha": [0.001, 0.01, 0.1, 1]},
    "Ridge": {"alpha": [0.01, 0.1, 1, 10, 100]},
    "Elastic Net": {"alpha": [0.001, 0.01, 0.1, 1], "l1_ratio": [0.1, 0.5, 0.9]},
    "Random Forest": {"n_estimators": [100, 200, 300], "max_depth": [None, 10, 20],
                       "max_features": [0.3, "sqrt", "log2"]},
    "SVM (SVR)": {"C": [0.1, 1, 10, 100], "gamma": ["scale", "auto", 0.01, 0.1], "epsilon": [0.01, 0.1, 0.5]},
    "Decision Tree": {"max_depth": [None, 3, 5, 10, 20]},
    "XGBoost": {"n_estimators": [100, 200, 300], "max_depth": [3, 5, 7], "learning_rate": [0.05, 0.1, 0.3]},
    "KNN": {"n_neighbors": [3, 5, 7, 9, 11]},
    "Neural Network (MLP)": {"hidden_layer_sizes": [(50,), (100,), (50, 50)], "alpha": [0.0001, 0.001, 0.01]},
}

# Algoritmos cuya optimización de hiperparámetros es notablemente más lenta
# (muchos árboles/estimadores por combinación). Se usa para avisar al usuario
# antes de que dispare un entrenamiento que puede tardar varios minutos.
ALGORITMOS_LENTOS_AL_OPTIMIZAR = {"Random Forest", "XGBoost", "SVM", "SVM (SVR)", "Neural Network (MLP)"}



def construir_particionador(cv, y, es_clasificacion, random_state=0):
    """
    Arma el particionador de validación cruzada a partir de 'cv', que puede
    ser un entero (k-fold) o el string "LOO" (Leave-One-Out: en cada
    repetición se deja UNA sola muestra afuera y se entrena con todo el
    resto — el caso extremo de k-fold con k = número de muestras).
    Devuelve (particionador, cantidad_de_repeticiones_usadas).
    """
    if isinstance(cv, str) and cv.upper() == "LOO":
        return LeaveOneOut(), len(y)
    if es_clasificacion:
        cv_real = min(cv, np.min(np.unique(y, return_counts=True)[1]))
        return StratifiedKFold(n_splits=cv_real, shuffle=True, random_state=random_state), cv_real
    cv_real = min(cv, len(y))
    return KFold(n_splits=cv_real, shuffle=True, random_state=random_state), cv_real


def _elegir_recurso_halving(grilla):
    """
    For Halving searches, decide which parameter to use as the "resource"
    that grows every round. If the model's grid has 'n_estimators' (Random
    Forest, XGBoost), use that as the resource — it fits how those models
    actually work (start with few trees, add more to the survivors). Every
    other model falls back to scikit-learn's default resource: n_samples.
    """
    if "n_estimators" in grilla:
        valores = sorted(grilla["n_estimators"])
        grilla_sin_recurso = {k: v for k, v in grilla.items() if k != "n_estimators"}
        return "n_estimators", grilla_sin_recurso, min(valores), max(valores)
    return "n_samples", grilla, None, None


def optimizar_hiperparametros(modelo, grilla, X, y, es_clasificacion, cv=5,
                               metodo="balanced", n_iter=20, random_state=0):
    """
    Searches for the best hyperparameters via cross-validation, using one of
    three speed/quality presets:

      - "fast": bounded random search (RandomizedSearchCV, few combinations
        tried). The fastest option; may miss the exact optimum.
      - "balanced": successive-halving search starting from random
        combinations (HalvingRandomSearchCV) — quickly discards unpromising
        combinations and only spends full effort on the best ones.
      - "thorough": successive-halving search over the FULL grid
        (HalvingGridSearchCV) — still evaluates every combination in the
        grid, but far more efficiently than classic grid search.

    ("grid" and "random" are kept as legacy aliases for plain GridSearchCV /
    RandomizedSearchCV, in case older code still calls them directly.)

    Returns (best_unfitted_model_with_those_params, best_params, best_score,
    method_description) — the description is meant to be shown to the user
    and stored in reports / the model card for traceability.
    """
    if not grilla:
        modelo.fit(X, y)
        return modelo, {}, None, "None (this model has no tunable hyperparameters)"

    scoring = "balanced_accuracy" if es_clasificacion else "r2"
    splitter, cv_real = construir_particionador(cv, y, es_clasificacion, random_state)
    n_muestras = X.shape[0]
    # Successive halving needs enough samples to shrink across rounds in a
    # meaningful way; below this size it degrades to a fallback plain search.
    UMBRAL_MUESTRAS_HALVING = 40
    es_loo = isinstance(cv, str) and cv.upper() == "LOO"

    if metodo == "grid":
        buscador = GridSearchCV(modelo, grilla, scoring=scoring, cv=splitter, n_jobs=N_JOBS)
        descripcion = "Grid search (exhaustive, legacy option)"

    elif metodo == "random":
        buscador = RandomizedSearchCV(
            modelo, grilla, scoring=scoring, cv=splitter, n_iter=n_iter,
            random_state=random_state, n_jobs=N_JOBS,
        )
        descripcion = f"Random search ({n_iter} combinations, legacy option)"

    elif metodo == "fast":
        n_iter_real = min(n_iter, 8) if n_iter else 8
        buscador = RandomizedSearchCV(
            modelo, grilla, scoring=scoring, cv=splitter, n_iter=n_iter_real,
            random_state=random_state, n_jobs=N_JOBS,
        )
        descripcion = f"Fast (random search, {n_iter_real} combinations tried)"

    elif metodo == "thorough":
        recurso_probable, _, _, _ = _elegir_recurso_halving(grilla)
        # HalvingSearchCV's resource-budget heuristics (min_resources based on
        # n_splits) don't work with LOO when the resource is n_samples: LOO
        # has as many splits as samples, which pushes the computed minimum
        # resource far above the total sample count and raises a ValueError.
        # n_estimators-based halving is unaffected (it never touches sample
        # counts), so only fall back for that specific combination.
        if n_muestras < UMBRAL_MUESTRAS_HALVING or (es_loo and recurso_probable == "n_samples"):
            buscador = GridSearchCV(modelo, grilla, scoring=scoring, cv=splitter, n_jobs=N_JOBS)
            motivo = "dataset too small for successive halving" if n_muestras < UMBRAL_MUESTRAS_HALVING \
                else "successive halving isn't compatible with LOO for this model"
            descripcion = f"Thorough (full grid search — {motivo})"
        else:
            recurso, grilla_reducida, min_r, max_r = _elegir_recurso_halving(grilla)
            kwargs = dict(estimator=modelo, param_grid=grilla_reducida, scoring=scoring,
                          cv=splitter, factor=3, resource=recurso, random_state=random_state, n_jobs=N_JOBS)
            if recurso == "n_estimators":
                kwargs["min_resources"] = min_r
                kwargs["max_resources"] = max_r
            buscador = HalvingGridSearchCV(**kwargs)
            descripcion = f"Thorough (successive halving grid search, resource={recurso})"

    elif metodo == "bayesian":
        # Bayesian optimization via Optuna (TPE sampler): unlike grid/random
        # search, each new combination is chosen using what was learned from
        # every PREVIOUS combination tried — so the search homes in on
        # promising regions instead of sampling blindly. Works with the same
        # discrete grids used everywhere else in the app (each hyperparameter
        # is treated as a categorical choice among the listed values).
        n_pruebas = max(10, min(n_iter, 40))

        def _objetivo(trial):
            params = {
                nombre: trial.suggest_categorical(nombre, [str(v) for v in valores])
                for nombre, valores in grilla.items()
            }
            # suggest_categorical needs hashable choices; strings round-trip
            # back to the original value (int/float/tuple/None) here.
            mapa_valores = {nombre: {str(v): v for v in valores} for nombre, valores in grilla.items()}
            params_reales = {k: mapa_valores[k][v] for k, v in params.items()}
            modelo_prueba = clone(modelo).set_params(**params_reales)
            puntajes = cross_val_score(modelo_prueba, X, y, cv=splitter, scoring=scoring, n_jobs=N_JOBS)
            return float(np.mean(puntajes))

        estudio = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=random_state),
        )
        estudio.optimize(_objetivo, n_trials=n_pruebas, show_progress_bar=False)

        mapa_valores = {nombre: {str(v): v for v in valores} for nombre, valores in grilla.items()}
        mejores_params = {k: mapa_valores[k][v] for k, v in estudio.best_params.items()}
        modelo_final = clone(modelo).set_params(**mejores_params)
        modelo_final.fit(X, y)
        return modelo_final, mejores_params, estudio.best_value, \
            f"Bayesian (Optuna/TPE, {n_pruebas} trials)"

    else:  # "balanced" (default preset)
        recurso_probable, _, _, _ = _elegir_recurso_halving(grilla)
        if n_muestras < UMBRAL_MUESTRAS_HALVING or (es_loo and recurso_probable == "n_samples"):
            buscador = RandomizedSearchCV(
                modelo, grilla, scoring=scoring, cv=splitter, n_iter=15,
                random_state=random_state, n_jobs=N_JOBS,
            )
            motivo = "dataset too small for successive halving" if n_muestras < UMBRAL_MUESTRAS_HALVING \
                else "successive halving isn't compatible with LOO for this model"
            descripcion = f"Balanced (random search, 15 combinations — {motivo})"
        else:
            recurso, grilla_reducida, min_r, max_r = _elegir_recurso_halving(grilla)
            kwargs = dict(estimator=modelo, param_distributions=grilla_reducida, scoring=scoring,
                          cv=splitter, factor=3, resource=recurso, random_state=random_state, n_jobs=N_JOBS)
            if recurso == "n_estimators":
                kwargs["min_resources"] = min_r
                kwargs["max_resources"] = max_r
            buscador = HalvingRandomSearchCV(**kwargs)
            descripcion = f"Balanced (successive halving random search, resource={recurso})"

    buscador.fit(X, y)
    return buscador.best_estimator_, buscador.best_params_, buscador.best_score_, descripcion


# =============================================================================
# EVALUACIÓN POR VALIDACIÓN CRUZADA
# =============================================================================

def especificidad_macro(y_true, y_pred, labels=None):
    """
    Especificidad (tasa de verdaderos negativos) promediada entre clases,
    calculada uno-contra-el-resto a partir de la matriz de confusión.
    Complementa a la sensibilidad (= recall_macro), que ya se calculaba.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if labels is None:
        labels = np.unique(y_true)
    matriz = confusion_matrix(y_true, y_pred, labels=labels)
    total = matriz.sum()
    especificidades = []
    for i in range(len(labels)):
        TP = matriz[i, i]
        FN = matriz[i, :].sum() - TP
        FP = matriz[:, i].sum() - TP
        TN = total - TP - FN - FP
        especificidades.append(TN / (TN + FP) if (TN + FP) > 0 else 0.0)
    return float(np.mean(especificidades))


def evaluar_clasificacion(modelo, X, y, cv=5, random_state=0):
    """
    Evalúa un clasificador por validación cruzada (estratificada, o LOO si
    cv="LOO"), usando predicciones "fuera de bolsa" (cross_val_predict) para
    que las métricas reflejen desempeño honesto sobre datos no vistos por
    cada modelo parcial.
    """
    y = np.asarray(y)
    if not (isinstance(cv, str) and cv.upper() == "LOO"):
        cv_real_check = min(cv, np.min(np.unique(y, return_counts=True)[1]))
        if cv_real_check < 2:
            raise ValueError(
                "Alguna clase tiene muy pocas muestras para hacer validación "
                "cruzada (se necesitan al menos 2 muestras por clase)."
            )
    splitter, cv_real = construir_particionador(cv, y, es_clasificacion=True, random_state=random_state)
    y_pred = cross_val_predict(modelo, X, y, cv=splitter, n_jobs=N_JOBS)

    auc = None
    y_proba = None
    if hasattr(modelo, "predict_proba"):
        try:
            y_proba = cross_val_predict(modelo, X, y, cv=splitter, method="predict_proba", n_jobs=N_JOBS)
            clases_unicas = np.unique(y)
            if len(clases_unicas) == 2:
                auc = float(roc_auc_score((y == clases_unicas[1]).astype(int), y_proba[:, 1]))
            else:
                auc = float(roc_auc_score(y, y_proba, multi_class="ovr", average="macro", labels=clases_unicas))
        except Exception:
            auc = None

    return {
        "y_pred": y_pred,
        "accuracy": accuracy_score(y, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y, y_pred),
        "f1_macro": f1_score(y, y_pred, average="macro"),
        "precision_macro": precision_score(y, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y, y_pred, average="macro", zero_division=0),
        "sensibilidad_macro": recall_score(y, y_pred, average="macro", zero_division=0),
        "especificidad_macro": especificidad_macro(y, y_pred),
        "kappa": cohen_kappa_score(y, y_pred),
        "mcc": float(matthews_corrcoef(y, y_pred)),
        "auc": auc,
        "y_proba": y_proba,                       # out-of-fold probabilities (calibration for conformal prediction)
        "clases_proba": np.unique(y),
        "matriz_confusion": confusion_matrix(y, y_pred),
        "reporte_por_clase": classification_report(y, y_pred, output_dict=True, zero_division=0),
        "clases": np.unique(y),
        "cv_folds_usados": cv_real,
    }


def evaluar_regresion(modelo, X, y, cv=5, random_state=0):
    """Evalúa un regresor por validación cruzada (o LOO si cv="LOO"), con
    predicción fuera de bolsa."""
    y = np.asarray(y, dtype=float)
    splitter, cv_real = construir_particionador(cv, y, es_clasificacion=False, random_state=random_state)
    y_pred = cross_val_predict(modelo, X, y, cv=splitter, n_jobs=N_JOBS)

    rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
    q75, q25 = np.percentile(y, [75, 25])
    return {
        "y_pred": y_pred,
        "rmse_cv": rmse,
        "mae_cv": float(mean_absolute_error(y, y_pred)),
        "r2_cv": float(r2_score(y, y_pred)),
        # RPD: cuántas veces el desvío estándar de y es más grande que el
        # error del modelo. Regla de uso frecuente en NIR: RPD > 2 aceptable,
        # > 3 bueno. RPIQ es la versión robusta, usando el rango intercuartil.
        "rpd_cv": float(np.std(y, ddof=1) / rmse) if rmse > 0 else np.inf,
        "rpiq_cv": float((q75 - q25) / rmse) if rmse > 0 else np.inf,
        "cv_folds_usados": cv_real,
    }


# =============================================================================
# DIVISIÓN ENTRENAMIENTO / TEST Y EVALUACIÓN COMPLETA
# =============================================================================

def kennard_stone(X, n_seleccionar):
    """
    Kennard-Stone algorithm: iteratively selects the sample farthest (in
    Euclidean distance) from the ones already selected, so the chosen subset
    covers the X-space as evenly and broadly as possible. Commonly used in
    chemometrics to pick a representative calibration/training set instead
    of a random split — the training set then includes the "extreme"
    samples, which tends to give a more honest, stable validation on the
    remaining (more "interior") samples.
    Returns the indices of the selected samples, in selection order.
    """
    from scipy.spatial.distance import pdist, squareform
    n = X.shape[0]
    n_seleccionar = min(n_seleccionar, n)
    if n_seleccionar < 2:
        return np.arange(n_seleccionar)

    dist = squareform(pdist(X))
    return _seleccion_por_distancia_maxima(dist, n_seleccionar)


def spxy(X, y, n_seleccionar):
    """
    SPXY (Sample set Partitioning based on joint X-Y distances): the
    regression-specific extension of Kennard-Stone (Galvão et al., 2005).
    Plain Kennard-Stone only looks at spectral (X) distance, which covers
    the spectral space well but gives no guarantee that the reference (Y)
    range is also well represented in both sets — e.g. the calibration set
    could end up missing the low end of the concentration range even while
    covering the spectral variability broadly. SPXY combines a normalized
    X-distance with a normalized Y-distance (the difference in reference
    value), so the selected calibration set is spread out across BOTH the
    spectral space and the concentration range at once.
    Returns the indices of the selected samples, in selection order.
    """
    from scipy.spatial.distance import pdist, squareform
    n = X.shape[0]
    n_seleccionar = min(n_seleccionar, n)
    if n_seleccionar < 2:
        return np.arange(n_seleccionar)

    dist_x = squareform(pdist(X))
    y_col = np.asarray(y, dtype=float).reshape(-1, 1)
    dist_y = squareform(pdist(y_col))

    # Normalize each distance matrix by its own maximum, so X and Y
    # contribute comparably regardless of their original scales/units.
    max_x = dist_x.max()
    max_y = dist_y.max()
    dist_x_norm = dist_x / max_x if max_x > 0 else dist_x
    dist_y_norm = dist_y / max_y if max_y > 0 else dist_y
    dist_combinada = dist_x_norm + dist_y_norm

    return _seleccion_por_distancia_maxima(dist_combinada, n_seleccionar)


def _seleccion_por_distancia_maxima(dist, n_seleccionar):
    """Shared greedy 'farthest point' selection used by both Kennard-Stone
    and SPXY — only the distance matrix passed in differs between them."""
    n = dist.shape[0]
    i, j = np.unravel_index(np.argmax(dist), dist.shape)
    seleccionados = [int(i), int(j)]
    restantes = [k for k in range(n) if k not in seleccionados]

    while len(seleccionados) < n_seleccionar:
        # For every remaining candidate, its "closeness to the selected set"
        # is its distance to the NEAREST already-selected sample. We pick
        # whichever remaining candidate is farthest from the selected set
        # (i.e., the most under-represented region of the space).
        distancias_al_set = dist[np.ix_(restantes, seleccionados)].min(axis=1)
        idx_elegido = int(np.argmax(distancias_al_set))
        nuevo = restantes.pop(idx_elegido)
        seleccionados.append(nuevo)

    return np.array(seleccionados)


def dividir_train_test(X, y, ids, proporcion_test=0.2, es_clasificacion=True,
                        random_state=0, metodo_split="random"):
    """
    Returns (idx_train, idx_test).
    metodo_split:
      - "random": stratified random split (default).
      - "kennard_stone": for classification, plain Kennard-Stone (X-distance
        only) is run SEPARATELY within each class, so every class
        contributes its own proportional share to both train and test
        (otherwise a class made up mostly of "extreme"/outlying samples
        could end up entirely in training, with zero representation in the
        test set). For regression, SPXY is used instead of plain
        Kennard-Stone: it combines spectral (X) distance with a distance in
        the reference value (Y), so the training set is spread out across
        BOTH the spectral space AND the concentration range — plain
        Kennard-Stone alone gives no such guarantee for the Y range.
    """
    idx = np.arange(len(y))
    if metodo_split == "kennard_stone":
        if es_clasificacion:
            idx_train_partes, idx_test_partes = [], []
            for clase in np.unique(y):
                idx_clase = idx[y == clase]
                if len(idx_clase) < 2:
                    # Too few samples in this class to hold any out — keep it all in training.
                    idx_train_partes.append(idx_clase)
                    continue
                n_train_clase = max(1, int(round(len(idx_clase) * (1 - proporcion_test))))
                n_train_clase = min(n_train_clase, len(idx_clase) - 1) if proporcion_test > 0 else len(idx_clase)
                seleccion_local = kennard_stone(X[idx_clase], n_train_clase)
                idx_train_clase = idx_clase[seleccion_local]
                idx_test_clase = np.array([i for i in idx_clase if i not in set(idx_train_clase.tolist())])
                idx_train_partes.append(idx_train_clase)
                if len(idx_test_clase):
                    idx_test_partes.append(idx_test_clase)
            idx_train = np.concatenate(idx_train_partes)
            idx_test = np.concatenate(idx_test_partes) if idx_test_partes else np.array([], dtype=int)
            return idx_train, idx_test

        n_train = int(round(len(y) * (1 - proporcion_test)))
        idx_train = spxy(X, y, n_train)
        idx_test = np.array([i for i in idx if i not in set(idx_train.tolist())])
        return idx_train, idx_test

    estratos = y if es_clasificacion else None
    idx_train, idx_test = train_test_split(
        idx, test_size=proporcion_test, stratify=estratos, random_state=random_state
    )
    return idx_train, idx_test


def entrenar_evaluar_clasificacion(modelo, X, y, ids=None, cv=5, proporcion_test=0.0, random_state=0,
                                    metodo_split="random", indices_split=None):
    """
    Full pipeline for classification:
    - If proporcion_test > 0: sets aside an independent test set (never used
      to choose anything), cross-validates ONLY on the rest (train), and
      finally trains on the full train set and evaluates once on the test set.
    - If proporcion_test == 0: cross-validates on 100% of the data (no
      independent test set left; useful when there are few samples).
    Returns a dict with 'cv' (always), 'test' (if applicable) and
    'modelo_final' (trained on the full train set, ready to save/predict).
    """
    X, y = np.asarray(X), np.asarray(y)
    ids = np.asarray(ids) if ids is not None else np.arange(len(y)).astype(str)

    if indices_split is not None:
        # explicit (train, test) split decided BEFORE variable selection / hyperparameter
        # optimization, so those steps could be fitted on the training samples only
        idx_train, idx_test = np.asarray(indices_split[0]), np.asarray(indices_split[1])
    elif proporcion_test > 0:
        idx_train, idx_test = dividir_train_test(X, y, ids, proporcion_test, True, random_state, metodo_split)
    else:
        idx_train, idx_test = np.arange(len(y)), None

    resultado = {"cv": evaluar_clasificacion(modelo, X[idx_train], y[idx_train], cv=cv, random_state=random_state)}
    resultado["cv"]["ids"] = ids[idx_train]
    resultado["cv"]["y_true"] = y[idx_train]

    modelo_final = clone(modelo)
    modelo_final.fit(X[idx_train], y[idx_train])
    resultado["modelo_final"] = modelo_final

    if idx_test is not None:
        y_pred_test = modelo_final.predict(X[idx_test])
        auc_test = None
        if hasattr(modelo_final, "predict_proba"):
            try:
                proba_test = modelo_final.predict_proba(X[idx_test])
                clases_unicas = np.unique(y)
                if len(clases_unicas) == 2:
                    idx_clase_pos = list(modelo_final.classes_).index(clases_unicas[1])
                    auc_test = float(roc_auc_score((y[idx_test] == clases_unicas[1]).astype(int),
                                                    proba_test[:, idx_clase_pos]))
                else:
                    auc_test = float(roc_auc_score(y[idx_test], proba_test, multi_class="ovr",
                                                    average="macro", labels=modelo_final.classes_))
            except Exception:
                auc_test = None
        resultado["test"] = {
            "ids": ids[idx_test],
            "y_true": y[idx_test],
            "y_pred": y_pred_test,
            "accuracy": accuracy_score(y[idx_test], y_pred_test),
            "balanced_accuracy": balanced_accuracy_score(y[idx_test], y_pred_test),
            "f1_macro": f1_score(y[idx_test], y_pred_test, average="macro"),
            "sensibilidad_macro": recall_score(y[idx_test], y_pred_test, average="macro", zero_division=0),
            "especificidad_macro": especificidad_macro(y[idx_test], y_pred_test),
            "kappa": cohen_kappa_score(y[idx_test], y_pred_test),
            "mcc": float(matthews_corrcoef(y[idx_test], y_pred_test)),
            "auc": auc_test,
            "matriz_confusion": confusion_matrix(y[idx_test], y_pred_test),
        }
    return resultado


def entrenar_evaluar_regresion(modelo, X, y, ids=None, cv=5, proporcion_test=0.0, random_state=0,
                                metodo_split="random", indices_split=None):
    """Same as entrenar_evaluar_clasificacion, but for regression (split not stratified)."""
    X, y = np.asarray(X), np.asarray(y, dtype=float)
    ids = np.asarray(ids) if ids is not None else np.arange(len(y)).astype(str)

    if indices_split is not None:
        # explicit (train, test) split decided BEFORE variable selection / hyperparameter
        # optimization, so those steps could be fitted on the training samples only
        idx_train, idx_test = np.asarray(indices_split[0]), np.asarray(indices_split[1])
    elif proporcion_test > 0:
        idx_train, idx_test = dividir_train_test(X, y, ids, proporcion_test, False, random_state, metodo_split)
    else:
        idx_train, idx_test = np.arange(len(y)), None

    resultado = {"cv": evaluar_regresion(modelo, X[idx_train], y[idx_train], cv=cv, random_state=random_state)}
    resultado["cv"]["ids"] = ids[idx_train]
    resultado["cv"]["y_true"] = y[idx_train]

    modelo_final = clone(modelo)
    modelo_final.fit(X[idx_train], y[idx_train])
    resultado["modelo_final"] = modelo_final

    if idx_test is not None:
        y_pred_test = modelo_final.predict(X[idx_test])
        rmse_test = float(np.sqrt(mean_squared_error(y[idx_test], y_pred_test)))
        resultado["test"] = {
            "ids": ids[idx_test],
            "y_true": y[idx_test],
            "y_pred": y_pred_test,
            "rmse": rmse_test,
            "mae": float(mean_absolute_error(y[idx_test], y_pred_test)),
            "r2": float(r2_score(y[idx_test], y_pred_test)),
            "rpd": float(np.std(y[idx_test], ddof=1) / rmse_test) if rmse_test > 0 else np.inf,
        }
    return resultado


# =============================================================================
# EXPORTACIÓN DE RESULTADOS
# =============================================================================

def exportar_predicciones_clasificacion(ids, y_true, y_pred):
    return pd.DataFrame({"id": ids, "true_class": y_true, "predicted_class": y_pred})


def exportar_predicciones_regresion(ids, y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return pd.DataFrame({
        "id": ids, "true_value": y_true, "predicted_value": y_pred, "residual": y_true - y_pred,
    })


# =============================================================================
# SELECCIÓN DE VARIABLES: BORUTA
# =============================================================================

def seleccionar_variables_boruta(X, y, es_clasificacion, random_state=0, max_iter=50, alpha=0.05):
    """
    Selección de variables con Boruta: compara la importancia de cada
    variable real contra copias aleatorizadas de sí mismas ("variables
    sombra"), y se queda con las que superan consistentemente a su versión
    aleatoria. Devuelve una máscara booleana (True = variable seleccionada).

    max_iter: cuántas rondas de comparación como máximo se corren (más
    rondas = decisión más estable, pero más lento).
    alpha: nivel de significancia del test estadístico interno que decide
    si una variable gana o pierde contra su sombra en cada ronda (más chico
    = más exigente para aceptar una variable como relevante).
    """
    from boruta import BorutaPy

    # Boruta recommends shallow trees; "auto" n_estimators and full-depth trees on
    # hundreds of spectral variables were the cause of multi-minute runs.
    kw = dict(n_estimators=100, max_depth=7, max_features="sqrt",
              random_state=random_state, n_jobs=N_JOBS)
    estimador = RandomForestClassifier(**kw) if es_clasificacion else RandomForestRegressor(**kw)

    seleccionador = BorutaPy(
        estimador, n_estimators=100, perc=90, random_state=random_state, max_iter=max_iter,
        alpha=alpha, verbose=0,
    )
    seleccionador.fit(np.asarray(X, dtype=float), np.asarray(y))
    mascara = np.asarray(seleccionador.support_, dtype=bool)
    if mascara.sum() < 2:
        # With hundreds of correlated variables few get formally "confirmed" in a short
        # run: fall back to confirmed + tentative ones instead of discarding the result.
        mascara = mascara | np.asarray(seleccionador.support_weak_, dtype=bool)
    if mascara.sum() < 2:
        mascara = np.zeros_like(mascara)      # callers treat 'nothing selected' as 'use all variables'
    return mascara


# =============================================================================
# SELECCIÓN DE VARIABLES: ALGORITMO GENÉTICO
# =============================================================================

class SeleccionGenetica:
    """
    Selección de variables por algoritmo genético: cada "individuo" es un
    subconjunto de variables (codificado como un vector binario), y se busca,
    generación tras generación, el subconjunto que mejor desempeño da en
    validación cruzada con un modelo evaluador.

    Pasos por generación: evaluar fitness de toda la población -> selección
    por torneo -> cruza uniforme -> mutación bit a bit -> elitismo (el mejor
    individuo siempre pasa a la siguiente generación sin cambios).
    """

    def __init__(self, modelo_evaluador, es_clasificacion, tam_poblacion=30,
                 n_generaciones=20, prob_mutacion=0.02, prob_cruza=0.8,
                 cv=3, min_variables=2, penalizacion_parsimonia=0.001, random_state=0):
        self.modelo_evaluador = modelo_evaluador
        self.es_clasificacion = es_clasificacion
        self.tam_poblacion = tam_poblacion
        self.n_generaciones = n_generaciones
        self.prob_mutacion = prob_mutacion
        self.prob_cruza = prob_cruza
        self.cv = cv
        self.min_variables = min_variables
        self.penalizacion_parsimonia = penalizacion_parsimonia
        self.random_state = random_state

    def _fitness(self, cromosoma, X, y):
        idx = np.where(cromosoma)[0]
        if len(idx) < self.min_variables:
            return -np.inf
        X_sub = X[:, idx]
        try:
            if self.es_clasificacion:
                cv_real = min(self.cv, np.min(np.unique(y, return_counts=True)[1]))
                if cv_real < 2:
                    return -np.inf
                skf = StratifiedKFold(n_splits=cv_real, shuffle=True, random_state=self.random_state)
                y_pred = cross_val_predict(self.modelo_evaluador, X_sub, y, cv=skf, n_jobs=N_JOBS)
                desempeno = balanced_accuracy_score(y, y_pred)
            else:
                cv_real = min(self.cv, X_sub.shape[0])
                kf = KFold(n_splits=cv_real, shuffle=True, random_state=self.random_state)
                y_pred = cross_val_predict(self.modelo_evaluador, X_sub, y, cv=kf, n_jobs=N_JOBS)
                desempeno = r2_score(y, y_pred)
        except Exception:
            return -np.inf

        # Penalización de parsimonia: entre dos subconjuntos con desempeño
        # similar, se prefiere el que usa menos variables (evita que el GA
        # "amontone" variables de más cuando el problema ya está resuelto).
        return desempeno - self.penalizacion_parsimonia * len(idx)

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n_variables = X.shape[1]

        poblacion = rng.rand(self.tam_poblacion, n_variables) < 0.3  # arranca con ~30% de variables activas
        self.historial_fitness_ = []

        for _generacion in range(self.n_generaciones):
            fitness = np.array([self._fitness(ind, X, y) for ind in poblacion])
            self.historial_fitness_.append(fitness.max())

            orden = np.argsort(fitness)[::-1]
            mejor = poblacion[orden[0]].copy()

            nueva_poblacion = [mejor]  # elitismo
            while len(nueva_poblacion) < self.tam_poblacion:
                padre1 = self._torneo(poblacion, fitness, rng)
                padre2 = self._torneo(poblacion, fitness, rng)
                if rng.rand() < self.prob_cruza:
                    mascara_cruza = rng.rand(n_variables) < 0.5
                    hijo = np.where(mascara_cruza, padre1, padre2)
                else:
                    hijo = padre1.copy()
                mutacion = rng.rand(n_variables) < self.prob_mutacion
                hijo = np.where(mutacion, ~hijo, hijo)
                nueva_poblacion.append(hijo)

            poblacion = np.array(nueva_poblacion)

        fitness_final = np.array([self._fitness(ind, X, y) for ind in poblacion])
        self.mejor_mascara_ = poblacion[np.argmax(fitness_final)]
        self.mejor_fitness_ = fitness_final.max()
        return self

    @staticmethod
    def _torneo(poblacion, fitness, rng, k=3):
        idx = rng.choice(len(poblacion), size=k, replace=False)
        ganador = idx[np.argmax(fitness[idx])]
        return poblacion[ganador]


# =============================================================================
# TRAZABILIDAD: ID único y ficha del entorno de software
# =============================================================================

def generar_id_trazabilidad():
    """
    ID corto y único para identificar un modelo entrenado y poder
    relacionarlo después con el reporte/ficha PDF generado en el momento del
    entrenamiento (formato: AAAAMMDD-HHMMSS-xxxxxx).
    """
    marca_tiempo = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    sufijo = secrets.token_hex(3)
    return f"{marca_tiempo}-{sufijo}"


def info_entorno_software():
    """Versiones de las librerías usadas, para dejar constancia en la ficha del modelo."""
    info = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    try:
        import sklearn
        info["scikit-learn"] = sklearn.__version__
    except Exception:
        pass
    try:
        import scipy
        info["scipy"] = scipy.__version__
    except Exception:
        pass
    try:
        import xgboost
        info["xgboost"] = xgboost.__version__
    except Exception:
        pass
    try:
        import boruta
        info["boruta"] = getattr(boruta, "__version__", "n/d")
    except Exception:
        pass
    try:
        import streamlit
        info["streamlit"] = streamlit.__version__
    except Exception:
        pass
    return info


# =============================================================================
# LEARNING CURVES
# =============================================================================

def calcular_curva_aprendizaje(modelo, X, y, es_clasificacion, cv=5,
                                train_sizes=None, random_state=0):
    """
    Computes a learning curve: training and cross-validation score as a
    function of how many samples were used to train. Useful to diagnose
    whether a model would benefit from more samples (both curves still
    rising / far apart) or has plateaued (curves flat and close together —
    more data probably won't help much).
    """
    if train_sizes is None:
        train_sizes = np.linspace(0.2, 1.0, 5)
    scoring = "balanced_accuracy" if es_clasificacion else "r2"
    splitter, _ = construir_particionador(cv, y, es_clasificacion, random_state)

    train_sizes_abs, train_scores, val_scores = learning_curve(
        modelo, X, y, cv=splitter, scoring=scoring, train_sizes=train_sizes,
        n_jobs=N_JOBS, random_state=random_state,
    )
    return {
        "train_sizes": train_sizes_abs,
        "train_scores_mean": train_scores.mean(axis=1),
        "train_scores_std": train_scores.std(axis=1),
        "val_scores_mean": val_scores.mean(axis=1),
        "val_scores_std": val_scores.std(axis=1),
        "scoring": scoring,
    }


# =============================================================================
# STATISTICAL COMPARISON BETWEEN MODELS
# =============================================================================

def _mcnemar_test(y_true, pred_a, pred_b):
    """
    McNemar's test for comparing two classifiers' predictions on the SAME
    test set — the standard approach in the ML literature for this exact
    situation (Dietterich, 1998). Looks only at the samples where the two
    models disagreed, and checks whether one of them was wrong noticeably
    more often than the other on those disagreements. Returns a two-sided
    p-value; uses an exact binomial test when there are few disagreements,
    and the usual chi-square approximation (with continuity correction)
    otherwise.
    """
    from scipy import stats
    y_true = np.asarray(y_true)
    correcto_a = (np.asarray(pred_a) == y_true)
    correcto_b = (np.asarray(pred_b) == y_true)
    b = int(np.sum((~correcto_a) & correcto_b))   # A wrong, B right
    c = int(np.sum(correcto_a & (~correcto_b)))   # A right, B wrong
    n = b + c
    if n == 0:
        return 1.0
    if n < 25:
        return float(stats.binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue)
    estadistico = (abs(b - c) - 1) ** 2 / n
    return float(1 - stats.chi2.cdf(estadistico, df=1))


def comparar_modelos_en_test(resultados_dict, es_clasificacion):
    """
    Compares models using their predictions on the SAME held-out test set,
    paired sample by sample — this is what actually reveals overfitting: two
    models can look equally good in cross-validation (which never sees the
    test set) yet perform very differently on genuinely unseen data. This is
    the more appropriate comparison whenever an independent test set was
    configured.

    Classification: McNemar's test on correct/incorrect predictions per test
    sample. Regression: paired t-test on the per-sample squared errors.

    resultados_dict: {name: result dict from entrenar_evaluar_*}, each
    expected to contain a "test" sub-result (proporcion_test > 0).
    Returns (p_values_df, per_model_test_score_dict) — the score is accuracy
    for classification or R² for regression (both "higher is better"), used
    only to say which model of a significant pair is the better one.
    """
    from scipy import stats

    nombres = [n for n in resultados_dict if "test" in resultados_dict[n]]
    n = len(nombres)
    pvalores = pd.DataFrame(np.ones((n, n)), index=nombres, columns=nombres)
    puntajes = {}

    for nombre in nombres:
        test_res = resultados_dict[nombre]["test"]
        puntajes[nombre] = test_res["accuracy"] if es_clasificacion else test_res["r2"]

    for i in range(n):
        for j in range(i + 1, n):
            a, b = nombres[i], nombres[j]
            y_true = resultados_dict[a]["test"]["y_true"]
            if es_clasificacion:
                p = _mcnemar_test(y_true, resultados_dict[a]["test"]["y_pred"], resultados_dict[b]["test"]["y_pred"])
            else:
                err_a = (np.asarray(resultados_dict[a]["test"]["y_pred"], dtype=float) - np.asarray(y_true, dtype=float)) ** 2
                err_b = (np.asarray(resultados_dict[b]["test"]["y_pred"], dtype=float) - np.asarray(y_true, dtype=float)) ** 2
                if np.allclose(err_a, err_b):
                    p = 1.0
                else:
                    _, p = stats.ttest_rel(err_a, err_b)
            pvalores.iloc[i, j] = p
            pvalores.iloc[j, i] = p

    return pvalores, puntajes


def comparar_modelos_estadisticamente(modelos_dict, X, y, es_clasificacion, cv=5, random_state=0):
    """
    Fallback comparison for when there is NO independent test set: compares
    several (already-configured, unfitted) models by running proper per-fold
    cross-validation on the SAME folds for all of them, then running a paired
    t-test between every pair of models on their per-fold scores. Whenever a
    test set is available, prefer comparar_modelos_en_test instead — it can
    actually detect overfitting, which this CV-only comparison cannot (since
    it never touches data the models haven't influenced in some way).

    modelos_dict: {name: unfitted model with its final hyperparameters}
    Returns (p_values_df, per_fold_scores_dict). A small p-value (e.g. < 0.05)
    means the performance difference between that pair of models is unlikely
    to be due to chance alone, given this dataset and these folds.
    """
    from sklearn.model_selection import cross_val_score
    from scipy import stats

    scoring = "balanced_accuracy" if es_clasificacion else "r2"
    splitter, _ = construir_particionador(cv, y, es_clasificacion, random_state)

    puntajes = {}
    for nombre, modelo in modelos_dict.items():
        try:
            puntajes[nombre] = cross_val_score(modelo, X, y, cv=splitter, scoring=scoring, n_jobs=N_JOBS)
        except Exception:
            puntajes[nombre] = None

    nombres = [n for n in modelos_dict if puntajes[n] is not None]
    n = len(nombres)
    pvalores = pd.DataFrame(np.ones((n, n)), index=nombres, columns=nombres)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = puntajes[nombres[i]], puntajes[nombres[j]]
            try:
                if np.allclose(a, b):
                    p = 1.0
                else:
                    _, p = stats.ttest_rel(a, b)
            except Exception:
                p = np.nan
            pvalores.iloc[i, j] = p
            pvalores.iloc[j, i] = p

    return pvalores, {n: puntajes[n] for n in nombres}


def interpretar_comparacion_modelos(pvalores, puntajes, alpha=0.05):
    """
    Turns the p-value table from comparar_modelos_estadisticamente into a
    short list of plain-language findings, e.g. "Random Forest significantly
    outperforms KNN (p=0.0231)". If no pair reaches significance, returns a
    single explicit message saying so, instead of leaving the user to guess
    from a bare table of numbers.
    """
    medias = {nombre: float(np.mean(valores)) for nombre, valores in puntajes.items()}
    nombres = list(pvalores.columns)
    hallazgos = []
    for i in range(len(nombres)):
        for j in range(i + 1, len(nombres)):
            a, b = nombres[i], nombres[j]
            p = pvalores.loc[a, b]
            if pd.notna(p) and p < alpha:
                mejor, peor = (a, b) if medias[a] > medias[b] else (b, a)
                hallazgos.append(f"**{mejor}** significantly outperforms **{peor}** (p={p:.4f}).")

    if not hallazgos:
        return {
            "hay_diferencias": False,
            "mensajes": ["No statistically significant differences were found among these models "
                         f"(all pairwise p-values ≥ {alpha}). The numeric differences in the metrics "
                         "table could just be noise from this particular split into folds — consider "
                         "them roughly equivalent unless you have another reason to prefer one."],
        }
    return {"hay_diferencias": True, "mensajes": hallazgos}
