"""Data augmentation for spectra / tabular data, applied ONLY to the training part of each
cross-validation fold (so test samples never see simulated copies of themselves)."""
import numpy as np
from sklearn.base import BaseEstimator, clone


def aumentar(X, y, es_clf, n_copias=2, ruido=0.02, base=0.0, escala=0.0, desplazamiento=0.0, mixup=0.0,
             espectral=True, semilla=0):
    """Returns (X_aug, y_aug): n_copias simulated versions of every sample (originals NOT included)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    rng = np.random.default_rng(semilla)
    n, p = X.shape
    sd = X.std(axis=0)
    sd_g = float(np.median(sd[sd > 0])) if np.any(sd > 0) else 1.0
    sd = np.where(sd > 0, sd, sd_g)
    xs = np.arange(p, dtype=float)
    ramp = np.linspace(-1, 1, p)
    Xs, ys = [], []
    for _ in range(int(n_copias)):
        Xc = X.copy()
        yc = y.copy()
        if mixup > 0:
            if es_clf:
                pareja = np.empty(n, dtype=int)
                for c in np.unique(y):
                    idx = np.where(y == c)[0]
                    pareja[idx] = rng.choice(idx, size=len(idx))
                lam = rng.uniform(1 - mixup, 1.0, size=(n, 1))
                Xc = lam * Xc + (1 - lam) * X[pareja]
            else:
                pareja = rng.integers(0, n, size=n)
                lam = rng.uniform(1 - mixup, 1.0, size=(n, 1))
                Xc = lam * Xc + (1 - lam) * X[pareja]
                yc = lam[:, 0] * y.astype(float) + (1 - lam[:, 0]) * y[pareja].astype(float)
        if espectral and desplazamiento > 0:
            d = rng.uniform(-desplazamiento, desplazamiento, size=n)
            for i in range(n):
                Xc[i] = np.interp(xs + d[i], xs, Xc[i])
        if escala > 0:
            Xc = Xc * (1 + rng.normal(0, escala, size=(n, 1)))
        if espectral and base > 0:
            Xc = Xc + rng.normal(0, base * sd_g, size=(n, 1)) + rng.normal(0, base * sd_g, size=(n, 1)) * ramp[None, :]
        if ruido > 0:
            Xc = Xc + rng.normal(0, 1, size=Xc.shape) * (ruido * sd)[None, :]
        Xs.append(Xc)
        ys.append(yc)
    return np.vstack(Xs), np.concatenate(ys)


class AumentadorEstimador(BaseEstimator):
    """Wraps any scikit-learn estimator: in fit() it adds simulated samples built from the samples it
    is given (the training fold), then fits the wrapped estimator. predict/predict_proba are untouched."""

    def __init__(self, estimator=None, es_clf=True, n_copias=2, ruido=0.02, base=0.0, escala=0.0,
                 desplazamiento=0.0, mixup=0.0, espectral=True, semilla=0):
        self.estimator = estimator
        self.es_clf = es_clf
        self.n_copias = n_copias
        self.ruido = ruido
        self.base = base
        self.escala = escala
        self.desplazamiento = desplazamiento
        self.mixup = mixup
        self.espectral = espectral
        self.semilla = semilla

    @property
    def _estimator_type(self):
        return "classifier" if self.es_clf else "regressor"

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier" if self.es_clf else "regressor"
        return tags

    def fit(self, X, y, **kw):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        Xa, ya = aumentar(X, y, self.es_clf, self.n_copias, self.ruido, self.base, self.escala,
                          self.desplazamiento, self.mixup, self.espectral, self.semilla)
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(np.vstack([X, Xa]), np.concatenate([y, ya]), **kw)
        if self.es_clf:
            self.classes_ = getattr(self.estimator_, "classes_", np.unique(y))
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def decision_function(self, X):
        return self.estimator_.decision_function(X)

    def score(self, X, y):
        return self.estimator_.score(X, y)

    def __getattr__(self, nombre):
        # only called when normal lookup fails: delegate to the fitted (or raw) wrapped estimator
        if nombre.startswith("__") or nombre in ("estimator_", "estimator"):
            raise AttributeError(nombre)
        d = self.__dict__
        base = d.get("estimator_", d.get("estimator"))
        if base is None:
            raise AttributeError(nombre)
        return getattr(base, nombre)

    def descripcion(self):
        partes = [f"×{int(self.n_copias)} copies"]
        if self.mixup: partes.append(f"mixup {self.mixup:g}")
        if self.ruido: partes.append(f"noise {self.ruido:g}")
        if self.escala: partes.append(f"scale ±{self.escala:g}")
        if self.espectral and self.base: partes.append(f"baseline {self.base:g}")
        if self.espectral and self.desplazamiento: partes.append(f"shift ±{self.desplazamiento:g} pts")
        return ", ".join(partes)
