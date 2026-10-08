"""Multi-block data fusion: compare each block alone with low-, mid- and high-level fusion."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.base import BaseEstimator
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score, r2_score, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
import models_utils_en as mu


class SOPLS(BaseEstimator):
    """Sequential and Orthogonalised PLS for two blocks given as ONE matrix [A | B] (n_a = columns of A).
    1) PLS of y on block A -> scores T_A.   2) B is orthogonalised against T_A (only what A does NOT already say).
    3) PLS on the orthogonalised B explains what is left of y.   4) y is regressed on [T_A, T_B].
    Handles regression (numeric y) and classification (labels -> one-hot, argmax)."""
    def __init__(self, n_a=1, nc_a=3, nc_b=3, es_clf=False):
        self.n_a, self.nc_a, self.nc_b, self.es_clf = n_a, nc_a, nc_b, es_clf

    def _Y(self, y):
        if self.es_clf:
            self.classes_ = np.unique(y)
            return (np.asarray(y)[:, None] == self.classes_[None, :]).astype(float)
        return np.asarray(y, dtype=float).reshape(len(y), -1)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); A, B = X[:, :self.n_a], X[:, self.n_a:]
        Y = self._Y(y)
        self.ma_, self.mb_, self.my_ = A.mean(0), B.mean(0), Y.mean(0)
        Ac, Bc, Yc = A - self.ma_, B - self.mb_, Y - self.my_
        ka = int(max(1, min(self.nc_a, Ac.shape[0] - 2, Ac.shape[1])))
        self.pa_ = PLSRegression(n_components=ka, scale=False).fit(Ac, Yc)
        TA = self.pa_.transform(Ac)
        self.G_ = np.linalg.pinv(TA.T @ TA) @ TA.T @ Bc
        Bo = Bc - TA @ self.G_
        kb = int(max(1, min(self.nc_b, Bo.shape[0] - 2, Bo.shape[1])))
        self.pb_ = PLSRegression(n_components=kb, scale=False).fit(Bo, Yc - self.pa_.predict(Ac).reshape(Yc.shape))
        TB = self.pb_.transform(Bo)
        T = np.hstack([TA, TB])
        self.W_ = np.linalg.pinv(T.T @ T) @ T.T @ Yc
        return self

    def _scores(self, X):
        X = np.asarray(X, dtype=float)
        Ac, Bc = X[:, :self.n_a] - self.ma_, X[:, self.n_a:] - self.mb_
        TA = self.pa_.transform(Ac)
        return np.hstack([TA, self.pb_.transform(Bc - TA @ self.G_)])

    def _raw(self, X):
        return self._scores(X) @ self.W_ + self.my_

    def predict(self, X):
        R = self._raw(X)
        if self.es_clf:
            return self.classes_[np.argmax(R, axis=1)]
        return R.ravel() if R.shape[1] == 1 else R


def preparar_bloque(X, modo="auto"):
    """modo: none | center | auto | pareto | snv (row-wise SNV followed by mean centering)"""
    X = np.asarray(X, dtype=float)
    if modo == "snv":
        X = (X - X.mean(axis=1, keepdims=True)) / np.where(X.std(axis=1, keepdims=True) > 0, X.std(axis=1, keepdims=True), 1)
        modo = "center"
    mu_ = X.mean(axis=0)
    Xc = X - mu_
    if modo == "none":
        return X
    if modo == "center":
        return Xc
    sd = X.std(axis=0, ddof=1)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return Xc / (sd if modo == "auto" else np.sqrt(sd))


def normalizar_bloque(X):
    """Divide a block by its Frobenius norm x sqrt(n): every block gets the same total variance, so a block with
    many variables (a spectrum) does not drown a block with few (a handful of compounds)."""
    f = np.linalg.norm(X) / np.sqrt(max(X.shape[0], 1))
    return X / (f if f > 0 else 1.0)


def _modelo(es_clf, n_comp):
    cat = mu.crear_clasificadores(n_componentes_pls=n_comp) if es_clf else mu.crear_regresores(n_componentes_pls=n_comp)
    return clone(cat["PLS-DA" if es_clf else "PLS"])


def _metrica(y, pred, es_clf):
    if es_clf:
        return float(balanced_accuracy_score(y, pred))
    return float(r2_score(y, pred)), float(np.sqrt(mean_squared_error(y, pred)))


def evaluar_fusion(XA, XB, y, es_clf, n_comp=5, k_pcs=5, cv=5, semilla=0):
    """XA, XB: already scaled blocks (same samples, same order). Returns a DataFrame of cross-validated results
    and the objects needed for plots."""
    n = len(y)
    if es_clf:
        cvs = StratifiedKFold(n_splits=int(min(cv, pd.Series(y).value_counts().min())), shuffle=True, random_state=semilla)
    else:
        cvs = KFold(n_splits=cv, shuffle=True, random_state=semilla)
    nc = int(max(1, min(n_comp, n - 2, XA.shape[1], XB.shape[1])))
    filas, preds, probas = [], {}, {}
    XAn, XBn = normalizar_bloque(XA), normalizar_bloque(XB)
    XL = np.hstack([XAn, XBn])
    k = int(max(1, min(k_pcs, n - 2)))
    # mid level: PCs of each (normalised) block (computed on all samples: unsupervised, no use of y)
    TA = PCA(n_components=min(k, XA.shape[1])).fit_transform(XAn)
    TB = PCA(n_components=min(k, XB.shape[1])).fit_transform(XBn)
    TM = np.hstack([TA / TA.std(), TB / TB.std()])
    esquemas = [("Block A alone", XAn), ("Block B alone", XBn), ("Low-level = MB-PLS (block-scaled, concatenated)", XL),
                ("Mid-level (PCs of each block)", TM)]
    for nombre, M in esquemas:
        m = _modelo(es_clf, int(min(nc, M.shape[1])))
        p = cross_val_predict(m, M, y, cv=cvs)
        preds[nombre] = p
        if es_clf and hasattr(m, "predict_proba"):
            try:
                probas[nombre] = cross_val_predict(m, M, y, cv=cvs, method="predict_proba")
            except Exception:
                pass
        filas.append((nombre, p))
    # sequential (SO-PLS): A first then B, and B first then A
    so = {}
    for nom, M, na, nca, ncb in [("SO-PLS (A first, then what B adds)", XL, XAn.shape[1], nc, max(1, min(3, nc))),
                                 ("SO-PLS (B first, then what A adds)", np.hstack([XBn, XAn]), XBn.shape[1], nc, max(1, min(3, nc)))]:
        try:
            p = cross_val_predict(SOPLS(n_a=na, nc_a=nca, nc_b=ncb, es_clf=bool(es_clf)), M, y, cv=cvs)
            preds[nom] = p; filas.append((nom, p))
        except Exception:
            pass
    # high level: combine the predictions of the two single-block models
    if es_clf and "Block A alone" in probas and "Block B alone" in probas:
        cl = np.unique(y)
        pm = (probas["Block A alone"] + probas["Block B alone"]) / 2
        ph = cl[np.argmax(pm, axis=1)]
        filas.append(("High-level (average of the two models)", ph))
    elif not es_clf:
        ph = (preds["Block A alone"] + preds["Block B alone"]) / 2
        filas.append(("High-level (average of the two models)", ph))
    out = []
    for nombre, p in filas:
        if es_clf:
            out.append({"Strategy": nombre, "CV balanced accuracy": round(_metrica(y, p, True), 3)})
        else:
            r2, rm = _metrica(y, p, False)
            out.append({"Strategy": nombre, "CV Q²": round(r2, 3), "RMSECV": round(rm, 4)})
    # block contribution in the low-level model: share of |coefficient| in each block
    m = _modelo(es_clf, nc).fit(XL, y)
    coef = np.abs(np.ravel(getattr(m, "coef_", np.zeros(XL.shape[1]))))
    if coef.size != XL.shape[1]:
        coef = np.abs(np.asarray(m.coef_)).sum(axis=tuple(range(np.asarray(m.coef_).ndim - 1))) if np.asarray(m.coef_).ndim > 1 else coef
    sA = float(coef[:XA.shape[1]].sum()); sB = float(coef[XA.shape[1]:].sum())
    contrib = (sA / (sA + sB) if sA + sB > 0 else 0.5)
    fus = PCA(n_components=2).fit(XL)
    return pd.DataFrame(out), {"contrib_A": contrib, "scores": fus.transform(XL), "var": fus.explained_variance_ratio_}
