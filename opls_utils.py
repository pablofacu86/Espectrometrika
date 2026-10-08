"""OPLS / OPLS-DA (Trygg & Wold 2002) for ONE response: the variation in X that is unrelated to y
(orthogonal) is removed and shown separately, leaving a single, interpretable predictive component."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold


class OPLS(BaseEstimator, RegressorMixin):
    def __init__(self, n_orth=1):
        self.n_orth = n_orth

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float).ravel()
        self.mx_, self.my_ = X.mean(0), y.mean()
        E, yc = X - self.mx_, y - self.my_
        W_o, P_o, T_o = [], [], []
        w = E.T @ yc; w /= (np.linalg.norm(w) or 1.0)
        for _ in range(int(self.n_orth)):
            t = E @ w
            p = E.T @ t / (t @ t)
            wo = p - (w @ p) / (w @ w) * w
            nw = np.linalg.norm(wo)
            if nw < 1e-12:
                break
            wo /= nw
            to = E @ wo
            po = E.T @ to / (to @ to)
            E = E - np.outer(to, po)
            W_o.append(wo); P_o.append(po); T_o.append(to)
            w = E.T @ yc; w /= (np.linalg.norm(w) or 1.0)
        self.Wo_ = np.array(W_o).reshape(len(W_o), X.shape[1]); self.Po_ = np.array(P_o).reshape(len(P_o), X.shape[1])
        self.To_ = np.array(T_o).T if T_o else np.zeros((len(y), 0))
        t = E @ w
        self.w_ = w
        self.q_ = float(t @ yc / (t @ t)) if t @ t > 0 else 0.0
        self.t_ = t
        self.p_ = E.T @ t / (t @ t) if t @ t > 0 else np.zeros(X.shape[1])
        self.E_ = E
        return self

    def _filter(self, X):
        E = np.asarray(X, dtype=float) - self.mx_
        for wo, po in zip(self.Wo_, self.Po_):
            E = E - np.outer(E @ wo, po)
        return E

    def transform(self, X):
        """Returns (predictive score, orthogonal scores) for X."""
        E = np.asarray(X, dtype=float) - self.mx_
        to = []
        for wo, po in zip(self.Wo_, self.Po_):
            t = E @ wo; to.append(t); E = E - np.outer(t, po)
        return E @ self.w_, (np.array(to).T if to else np.zeros((len(E), 0)))

    def predict(self, X):
        return self._filter(X) @ self.w_ * self.q_ + self.my_


def cv_orth(X, y, es_clf, max_orth=5, cv=7, semilla=0):
    """Q² (or CV balanced accuracy / AUC for 0-1 y) against the number of orthogonal components."""
    X = np.asarray(X, float); y = np.asarray(y, float)
    n = len(y)
    kf = (StratifiedKFold(min(cv, int(min(np.sum(y == 0), np.sum(y == 1)))), shuffle=True, random_state=semilla)
          if es_clf else KFold(min(cv, n), shuffle=True, random_state=semilla))
    filas = []
    for k in range(0, max_orth + 1):
        pred = np.zeros(n)
        for tr, te in kf.split(X, y):
            pred[te] = OPLS(k).fit(X[tr], y[tr]).predict(X[te])
        press = float(((y - pred) ** 2).sum()); tss = float(((y - y.mean()) ** 2).sum())
        r = {"orthogonal components": k, "Q²": round(1 - press / tss, 3) if tss > 0 else np.nan}
        if es_clf:
            r["AUC (CV)"] = round(float(roc_auc_score(y, pred)), 3)
            r["accuracy (CV)"] = round(float(((pred > 0.5) == (y > 0.5)).mean()), 3)
        r["R²Y (fit)"] = round(float(OPLS(k).fit(X, y).score(X, y)), 3)
        filas.append(r)
    return pd.DataFrame(filas)


def s_plot(m, X):
    """Covariance and correlation of every variable with the predictive score (loading profile of OPLS)."""
    X = np.asarray(X, float); Xc = X - X.mean(0); t = m.t_
    cov = Xc.T @ t / (len(t) - 1)
    sx = Xc.std(0, ddof=1); st = t.std(ddof=1)
    corr = np.where(sx > 1e-12, cov / (sx * st + 1e-30), 0.0)
    return cov, np.clip(corr, -1, 1)
