"""
Non-linear counterpart of the PC finder, working directly on the variables (no PCA):

* `ventanas`            contiguous windows along the axis (neighbouring spectral variables are highly
                        collinear, so importance is measured per window, not per single variable).
* `permutacion_ventanas` grouped permutation importance with a Random Forest, always on held-out samples
                        (out-of-fold): how much worse the predictions get when ONE window is scrambled.
                        Captures non-linear effects and interactions between regions.
* `mi_ventanas`         mutual information between each window's mean signal and the class / value
                        (non-linear, univariate).
* `comparar_ventanas`   side by side comparison of a linear and a non-linear importance (consensus).
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.model_selection import KFold, StratifiedKFold


def ventanas(p, n_ventanas):
    n_ventanas = int(max(2, min(n_ventanas, p)))
    cortes = np.linspace(0, p, n_ventanas + 1).astype(int)
    return [np.arange(cortes[i], cortes[i + 1]) for i in range(n_ventanas) if cortes[i + 1] > cortes[i]]


def _rf(es_clf, n_est, n_jobs=1):
    kw = dict(n_estimators=n_est, max_features="sqrt" if es_clf else 0.2, min_samples_leaf=1,
              random_state=0, n_jobs=n_jobs)
    return RandomForestClassifier(**kw) if es_clf else RandomForestRegressor(**kw)


def _error(modelo, X, y, es_clf, clases):
    """Smooth error: Brier score for classification, MSE for regression (lower = better)."""
    if es_clf:
        P = modelo.predict_proba(X)
        Y = (np.asarray(y)[:, None] == clases[None, :]).astype(float)
        P_full = np.zeros_like(Y)
        for j, c in enumerate(modelo.classes_):
            P_full[:, np.where(clases == c)[0][0]] = P[:, j]
        return float(((P_full - Y) ** 2).sum(axis=1).mean())
    return float(((modelo.predict(X) - y) ** 2).mean())


def permutacion_ventanas(X, y, es_clf, wins, n_est=150, n_rep=2, cv=5, n_jobs=1, seed=0):
    """Returns (importance per window >= 0 scaled to max 1, per-variable RF impurity importance,
    baseline out-of-fold score). Importance = increase of the held-out error when the window is
    permuted, averaged over folds and repetitions."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).astype(str) if es_clf else np.asarray(y, dtype=float)
    rng = np.random.RandomState(seed)
    clases = np.unique(y) if es_clf else None
    if es_clf:
        cv = int(min(cv, np.unique(y, return_counts=True)[1].min()))
        splitter = StratifiedKFold(n_splits=max(2, cv), shuffle=True, random_state=seed)
    else:
        splitter = KFold(n_splits=max(2, min(cv, len(y) // 3)), shuffle=True, random_state=seed)
    delta = np.zeros(len(wins))
    base_err, n_folds = 0.0, 0
    for tr, te in splitter.split(X, y if es_clf else None):
        rf = _rf(es_clf, n_est, n_jobs).fit(X[tr], y[tr])
        e0 = _error(rf, X[te], y[te], es_clf, clases)
        base_err += e0; n_folds += 1
        for k, w in enumerate(wins):
            acc = 0.0
            for _ in range(n_rep):
                Xp = X[te].copy()
                Xp[:, w] = Xp[rng.permutation(len(te))][:, w]
                acc += _error(rf, Xp, y[te], es_clf, clases) - e0
            delta[k] += acc / n_rep
    delta = np.clip(delta / max(n_folds, 1), 0, None)
    imp_w = delta / (delta.max() or 1.0)
    rf_all = _rf(es_clf, n_est, n_jobs).fit(X, y)
    imp_var = rf_all.feature_importances_
    imp_var = imp_var / (imp_var.max() or 1.0)
    return imp_w, imp_var, base_err / max(n_folds, 1)


def mi_ventanas(X, y, es_clf, wins, seed=0):
    M = np.column_stack([np.asarray(X, dtype=float)[:, w].mean(axis=1) for w in wins])
    if es_clf:
        mi = mutual_info_classif(M, np.asarray(y).astype(str), n_neighbors=3, random_state=seed)
    else:
        mi = mutual_info_regression(M, np.asarray(y, dtype=float), n_neighbors=3, random_state=seed)
    return mi / (mi.max() or 1.0)


def a_ventanas(imp_var, wins):
    """Mean of a per-variable curve inside each window."""
    return np.array([float(np.mean(imp_var[w])) for w in wins])


def comparar_ventanas(eje, wins, lineal_w, no_lineal_w, top_frac=0.2):
    """Consensus table per window: both methods, rank-based agreement and a category."""
    lineal_w = np.asarray(lineal_w, float); no_lineal_w = np.asarray(no_lineal_w, float)
    n = len(wins)
    k = max(1, int(round(top_frac * n)))
    top_l = set(np.argsort(lineal_w)[::-1][:k]); top_n = set(np.argsort(no_lineal_w)[::-1][:k])
    rho = spearmanr(lineal_w, no_lineal_w).statistic if n > 2 else np.nan
    jac = len(top_l & top_n) / max(1, len(top_l | top_n))
    rank_l = pd.Series(lineal_w).rank(pct=True).to_numpy(); rank_n = pd.Series(no_lineal_w).rank(pct=True).to_numpy()
    eje = np.asarray(eje, float)
    filas = []
    for i, w in enumerate(wins):
        cat = "Both" if (i in top_l and i in top_n) else ("Linear only" if i in top_l else
                                                           ("Non-linear only" if i in top_n else ""))
        filas.append({"From": float(eje[w[0]]), "To": float(eje[w[-1]]), "Linear": float(lineal_w[i]),
                      "Non-linear": float(no_lineal_w[i]), "Consensus (rank mean)": float((rank_l[i] + rank_n[i]) / 2),
                      "Found by": cat, "_i": i})
    df = pd.DataFrame(filas)
    return df, {"spearman": float(rho) if rho == rho else np.nan, "overlap": float(jac), "k": k}
