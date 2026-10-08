"""Instrument transfer lab: several datasets measured on different instruments -> cross-prediction matrix,
pre-treatments that reduce the between-instrument difference, pooled (mixed) training and enrichment curves."""
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

PRETRAT = {
    "none": "None (raw)",
    "snv": "SNV",
    "msc": "MSC (pooled reference)",
    "d1": "1st derivative (Savitzky-Golay)",
    "d2": "2nd derivative (Savitzky-Golay)",
    "snv_d1": "SNV + 1st derivative",
    "area": "Area normalisation",
    "center_inst": "Mean-centre each instrument",
    "proj_inst": "Remove instrument-mean directions (EPO-lite)",
}


# ------------------------------------------------------------------ reading / common grid
def leer_tabla(df):
    """First column = ID. Columns whose header is a number = spectrum; the others = reference/class columns."""
    ids = df.iloc[:, 0].astype(str).to_numpy()
    esp, otras = [], []
    for c in df.columns[1:]:
        try:
            float(str(c).replace(",", ".").strip()); esp.append(c)
        except ValueError:
            otras.append(c)
    if len(esp) < 10:
        raise ValueError("fewer than 10 numeric column headers: this does not look like a spectra table")
    eje = np.array([float(str(c).replace(",", ".").strip()) for c in esp])
    X = df[esp].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    o = np.argsort(eje)
    return ids, df[otras].copy(), eje[o], X[:, o]


def rango_comun(ejes):
    lo = max(float(e.min()) for e in ejes); hi = min(float(e.max()) for e in ejes)
    return lo, hi


def interpolar(X, eje, grid):
    out = np.empty((X.shape[0], len(grid)))
    for i in range(X.shape[0]):
        out[i] = np.interp(grid, eje, np.nan_to_num(X[i], nan=np.nanmean(X[i]) if np.isfinite(X[i]).any() else 0.0))
    return out


# ------------------------------------------------------------------ pre-treatments
def _snv(X):
    sd = X.std(axis=1, keepdims=True)
    return (X - X.mean(axis=1, keepdims=True)) / np.where(sd > 0, sd, 1)


def preprocesar(Xs, nombre, ventana=11):
    """Xs: list of matrices (one per instrument, same variables). Returns a list of the same length."""
    Xs = [np.asarray(x, float) for x in Xs]
    v = int(ventana) | 1
    if nombre == "none":
        return Xs
    if nombre == "snv":
        return [_snv(x) for x in Xs]
    if nombre == "msc":
        ref = np.vstack([x.mean(0) for x in Xs]).mean(0)
        out = []
        for x in Xs:
            o = np.empty_like(x)
            for i, s in enumerate(x):
                b, a = np.polyfit(ref, s, 1); o[i] = (s - a) / (b if abs(b) > 1e-12 else 1)
            out.append(o)
        return out
    if nombre == "d1":
        return [savgol_filter(x, v, 2, deriv=1, axis=1) for x in Xs]
    if nombre == "d2":
        return [savgol_filter(x, v, 3, deriv=2, axis=1) for x in Xs]
    if nombre == "snv_d1":
        return [savgol_filter(_snv(x), v, 2, deriv=1, axis=1) for x in Xs]
    if nombre == "area":
        return [x / np.where(np.abs(x).sum(axis=1, keepdims=True) > 0, np.abs(x).sum(axis=1, keepdims=True), 1) for x in Xs]
    if nombre == "center_inst":
        return [x - x.mean(0) for x in Xs]
    if nombre == "proj_inst":
        M = np.vstack([x.mean(0) for x in Xs]); D = M - M.mean(0)
        U, S, Vt = np.linalg.svd(D, full_matrices=False)
        k = int((S > 1e-9 * max(S.max(), 1e-30)).sum())
        V = Vt[:k]
        return [x - (x @ V.T) @ V for x in Xs]
    raise ValueError(nombre)


def distancia_instrumentos(Xs):
    """Mean RMS distance between the mean spectra of the instruments, relative to the within-instrument spread."""
    M = np.vstack([x.mean(0) for x in Xs])
    d = [np.sqrt(np.mean((M[i] - M[j]) ** 2)) for i in range(len(Xs)) for j in range(i + 1, len(Xs))]
    intra = np.mean([np.sqrt(np.mean(x.var(axis=0))) for x in Xs])
    return float(np.mean(d) / (intra if intra > 0 else 1.0)) if d else 0.0


# ------------------------------------------------------------------ models
class _PLS:
    def __init__(self, n_comp, es_clf):
        self.n, self.clf = int(n_comp), es_clf

    def fit(self, X, y):
        k = int(max(1, min(self.n, X.shape[0] - 2, X.shape[1])))
        if self.clf:
            self.cl = np.unique(y); Y = (np.asarray(y)[:, None] == self.cl[None, :]).astype(float)
        else:
            Y = np.asarray(y, float).reshape(-1, 1)
        self.m = PLSRegression(n_components=k, scale=False).fit(X, Y); return self

    def predict(self, X):
        p = self.m.predict(X)
        return self.cl[np.argmax(p, axis=1)] if self.clf else np.ravel(p)


def _metricas(y, p, es_clf):
    if es_clf:
        return {"metric": float((np.asarray(y) == np.asarray(p)).mean()), "bias": np.nan}
    y = np.asarray(y, float); p = np.asarray(p, float)
    return {"metric": float(np.sqrt(np.mean((y - p) ** 2))), "bias": float(np.mean(p - y))}


def matriz_cruzada(Xs, ys, nombres, es_clf, n_comp=6, cv=5, semilla=0):
    """Rows = trained on, columns = tested on. Diagonal = cross-validation inside the instrument."""
    n = len(Xs)
    M = np.full((n, n), np.nan); B = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if i == j:
                p = np.empty(len(ys[i]), dtype=ys[i].dtype)
                for tr, te in KFold(min(cv, len(ys[i])), shuffle=True, random_state=semilla).split(Xs[i]):
                    p[te] = _PLS(n_comp, es_clf).fit(Xs[i][tr], ys[i][tr]).predict(Xs[i][te])
            else:
                p = _PLS(n_comp, es_clf).fit(Xs[i], ys[i]).predict(Xs[j])
            r = _metricas(ys[j], p, es_clf); M[i, j], B[i, j] = r["metric"], r["bias"]
    return pd.DataFrame(M, index=nombres, columns=nombres), pd.DataFrame(B, index=nombres, columns=nombres)


def comparar_pretratamientos(Xs, ys, nombres, es_clf, n_comp=6, ventana=11, claves=None, cv=5):
    filas = []
    for k in (claves or list(PRETRAT)):
        try:
            Xp = preprocesar(Xs, k, ventana)
            M, B = matriz_cruzada(Xp, ys, nombres, es_clf, n_comp, cv)
            a = M.to_numpy(); off = a[~np.eye(len(a), dtype=bool)]; dg = np.diag(a)
            filas.append({"Pre-treatment": PRETRAT[k], "_k": k,
                          ("Accuracy between instruments" if es_clf else "RMSEP between instruments"): float(np.mean(off)),
                          ("Accuracy inside instrument (CV)" if es_clf else "RMSECV inside instrument"): float(np.mean(dg)),
                          "Instrument distance": distancia_instrumentos(Xp)})
        except Exception as e:
            filas.append({"Pre-treatment": PRETRAT[k], "_k": k, "note": str(e)[:60]})
    return pd.DataFrame(filas)


def entrenamiento_mezclado(Xs, ys, nombres, idx_train, idx_test, es_clf, n_comp=6):
    """Train with the pooled samples of the instruments idx_train, test on each instrument of idx_test."""
    Xt = np.vstack([Xs[i] for i in idx_train]); yt = np.concatenate([ys[i] for i in idx_train])
    m = _PLS(n_comp, es_clf).fit(Xt, yt)
    filas = []
    for j in idx_test:
        if j in idx_train:
            continue
        r = _metricas(ys[j], m.predict(Xs[j]), es_clf)
        filas.append({"Tested on": nombres[j], "Accuracy" if es_clf else "RMSEP": r["metric"], "Bias": r["bias"]})
    return pd.DataFrame(filas)


def curva_enriquecimiento(Xs, ys, nombres, idx_train, j_test, es_clf, n_comp=6, tamanos=(0, 3, 5, 10, 20), reps=5, semilla=0):
    """Pooled training + n samples of the TARGET instrument; evaluated on the remaining target samples."""
    rng = np.random.default_rng(semilla)
    Xb = np.vstack([Xs[i] for i in idx_train]); yb = np.concatenate([ys[i] for i in idx_train])
    nT = len(ys[j_test]); filas = []
    for k in tamanos:
        if k > nT - 5:
            continue
        vals = []
        for _ in range(reps if k > 0 else 1):
            sel = rng.permutation(nT); a, b = sel[:k], sel[k:]
            X = np.vstack([Xb, Xs[j_test][a]]) if k else Xb
            y = np.concatenate([yb, ys[j_test][a]]) if k else yb
            vals.append(_metricas(ys[j_test][b], _PLS(n_comp, es_clf).fit(X, y).predict(Xs[j_test][b]), es_clf)["metric"])
        filas.append({"Target samples added": k, "Accuracy" if es_clf else "RMSEP": float(np.mean(vals)), "sd": float(np.std(vals))})
    return pd.DataFrame(filas)
