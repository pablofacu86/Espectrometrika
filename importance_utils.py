"""Model-agnostic importance: grouped permutation of spectral windows, measured on samples (held-out if you have them)."""
import numpy as np
from sklearn.metrics import balanced_accuracy_score, r2_score


def _score(modelo, X, y, es_clf):
    p = modelo.predict(X)
    return float(balanced_accuracy_score(y, p)) if es_clf else float(r2_score(y, p))


def permutacion_ventanas(modelo, X, y, es_clf, n_ventanas=40, n_rep=5, semilla=0):
    """Returns (windows, drop, sd, base_score). drop = base score - score after shuffling that window's columns
    across samples (bigger drop = the model relies more on that region)."""
    X = np.asarray(X, float); y = np.asarray(y)
    p = X.shape[1]
    nv = int(max(2, min(n_ventanas, p)))
    wins = np.array_split(np.arange(p), nv)
    rng = np.random.default_rng(semilla)
    base = _score(modelo, X, y, es_clf)
    drop, sd = np.zeros(nv), np.zeros(nv)
    for k, w in enumerate(wins):
        v = []
        for _ in range(int(n_rep)):
            Xp = X.copy()
            Xp[:, w] = Xp[rng.permutation(len(X))][:, w]
            v.append(base - _score(modelo, Xp, y, es_clf))
        drop[k], sd[k] = float(np.mean(v)), float(np.std(v))
    return wins, drop, sd, base
