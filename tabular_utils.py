"""Preprocessing for tabular (non-spectral) variables: metabolomics, clinical data, etc.
Samples x variables matrices; no spectral axis needed. The preprocessor is FITTED on the
training samples (imputation values, offsets, means, scales) so that new samples can be
transformed with exactly the same parameters (Prediction tab)."""
import numpy as np


def _col_stat(X, f):
    with np.errstate(all="ignore"):
        v = f(X, axis=0)
    return np.where(np.isfinite(v), v, 0.0)


class Preprocesador:
    def __init__(self, imputacion="median", norm_filas="none", transformacion="none", escalado="auto"):
        self.imputacion, self.norm_filas = imputacion, norm_filas
        self.transformacion, self.escalado = transformacion, escalado

    # ---- fit ----------------------------------------------------------------
    def _fit_imputar(self, X):
        m = self.imputacion
        if m == "zero":
            self.v_imp = np.zeros(X.shape[1])
        elif m == "min2":
            Xm = np.where(X > 0, X, np.nan)
            self.v_imp = _col_stat(Xm, np.nanmin) / 2
        elif m == "mean":
            self.v_imp = _col_stat(X, np.nanmean)
        else:
            self.v_imp = _col_stat(X, np.nanmedian)

    def _imputar(self, X):
        X = np.array(X, dtype=float)
        i, j = np.where(np.isnan(X))
        if len(i):
            X[i, j] = self.v_imp[j]
        return X

    def _norm(self, X, ajustar):
        m = self.norm_filas
        if m == "none":
            return X
        if m == "sum":
            s = np.abs(X).sum(axis=1, keepdims=True)
            if ajustar:
                self.esc_norm = float(np.median(s))
            return X / np.where(s > 0, s, 1) * self.esc_norm
        if m == "median":
            s = np.median(X, axis=1, keepdims=True)
            if ajustar:
                self.esc_norm = float(np.median(s))
            return X / np.where(np.abs(s) > 1e-12, s, 1) * self.esc_norm
        if m == "pqn":
            if ajustar:
                self.ref_pqn = np.median(X, axis=0)
            ok = np.abs(self.ref_pqn) > 1e-12
            q = np.median(X[:, ok] / self.ref_pqn[ok], axis=1, keepdims=True)
            return X / np.where(np.abs(q) > 1e-12, q, 1)
        raise ValueError(m)

    def _transf(self, X, ajustar):
        m = self.transformacion
        if m == "none":
            return X
        if m == "sqrt":
            return np.sqrt(np.clip(X, 0, None))
        if m in ("log10", "log2", "ln"):
            if ajustar:
                mn = X.min(axis=0)
                self.off = np.where(mn <= 0, -mn + 1e-3 * (np.abs(X).mean(axis=0) + 1e-12) + 1e-9, 0.0)
            f = {"log10": np.log10, "log2": np.log2, "ln": np.log}[m]
            return f(np.clip(X + self.off, 1e-12, None))
        if m == "glog":
            if ajustar:
                self.lam = (np.median(np.abs(X)) * 1e-2 + 1e-12) ** 2
            return np.log((X + np.sqrt(X ** 2 + self.lam)) / 2)
        raise ValueError(m)

    def _escala(self, X, ajustar):
        m = self.escalado
        if m == "none":
            return X
        if ajustar:
            mu = X.mean(axis=0)
            sd = X.std(axis=0, ddof=1) if X.shape[0] > 1 else np.ones(X.shape[1])
            cen = mu
            if m == "center":
                d = np.ones(X.shape[1])
            elif m == "auto":
                d = sd
            elif m == "pareto":
                d = np.sqrt(sd)
            elif m == "range":
                d = X.max(axis=0) - X.min(axis=0)
            elif m == "level":
                d = np.where(np.abs(mu) > 1e-12, np.abs(mu), 1.0)
            elif m == "robust":
                cen = np.median(X, axis=0)
                d = 1.4826 * np.median(np.abs(X - cen), axis=0)
            else:
                raise ValueError(m)
            self.cen, self.div = cen, np.where(d > 1e-12, d, 1.0)
        return (X - self.cen) / self.div

    # ---- API ----------------------------------------------------------------
    def fit_transform(self, X):
        X = np.array(X, dtype=float)
        self._fit_imputar(X)
        X = self._imputar(X)
        X = self._norm(X, True)
        X = self._transf(X, True)
        return self._escala(X, True)

    def transform(self, X):
        X = self._imputar(np.array(X, dtype=float))
        X = self._norm(X, False)
        X = self._transf(X, False)
        return self._escala(X, False)

    def describir(self):
        n = {"median": "median imputation", "mean": "mean imputation", "min2": "half-minimum imputation",
             "zero": "zero imputation"}
        t = {"none": None, "log10": "log10", "ln": "ln", "log2": "log2", "sqrt": "square root", "glog": "glog"}
        e = {"none": None, "center": "mean centering", "auto": "autoscaling", "pareto": "Pareto scaling",
             "range": "range scaling", "level": "level scaling", "robust": "robust scaling (median/MAD)"}
        r = {"none": None, "sum": "row sum normalization", "median": "row median normalization", "pqn": "PQN"}
        partes = [n[self.imputacion], r[self.norm_filas], t[self.transformacion], e[self.escalado]]
        return " → ".join(x for x in partes if x)


def imputar(X, metodo="median"):
    p = Preprocesador(imputacion=metodo)
    p._fit_imputar(np.array(X, dtype=float))
    return p._imputar(X)
