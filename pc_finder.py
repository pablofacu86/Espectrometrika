"""
Search for the most informative combination of principal components, and trace it back to the
original variables (wavenumbers / ppm / retention times).

* Classification: which PCs (alone, in pairs or in triples) best separate the classes.
  Criterion = 1 - Wilks' lambda (multivariate eta²: fraction of the variance in those PCs that lies
  BETWEEN classes, 0-1; for a single PC it is the usual ANOVA eta²). The best combinations are then
  confirmed with a cross-validated LDA (balanced accuracy) on those PCs only.
* Regression: which PCs (alone, in pairs or in triples) best explain the reference values
  (adjusted R² of a linear fit, confirmed with a cross-validated Q²).
* Then the direction found in PC space is projected back onto the variables, giving a "variable
  importance" curve along the spectrum / chromatogram and the regions that carry the difference.
"""
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.ndimage import label, uniform_filter1d
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict


def ajustar_pca(X, n_comp):
    n_comp = int(max(1, min(n_comp, X.shape[0] - 1, X.shape[1])))
    pca = PCA(n_components=n_comp, svd_solver="full" if X.shape[1] < 3000 else "randomized", random_state=0)
    scores = pca.fit_transform(np.asarray(X, dtype=float))
    return scores, pca.components_, pca.explained_variance_ratio_ * 100


def _wilks_sep(S, y_idx, n_clases):
    """1 - Wilks' lambda for the columns of S (n x d) given integer class indices."""
    Sc = S - S.mean(axis=0)
    T = Sc.T @ Sc
    W = np.zeros_like(T)
    for c in range(n_clases):
        g = Sc[y_idx == c]
        if len(g):
            gc = g - g.mean(axis=0)
            W += gc.T @ gc
    sT = np.linalg.slogdet(T + 1e-12 * np.eye(T.shape[0]))[1]
    sW = np.linalg.slogdet(W + 1e-12 * np.eye(W.shape[0]))[1]
    return float(1.0 - np.exp(min(0.0, sW - sT)))


def buscar_clasificacion(scores, clases, k_max, tam, n_confirmar=12):
    clases = np.asarray(clases).astype(str)
    uniq, y_idx = np.unique(clases, return_inverse=True)
    k_max = min(k_max, scores.shape[1])
    filas = []
    for comb in combinations(range(k_max), tam):
        filas.append({"cols": comb, "Separation (1-Wilks)": _wilks_sep(scores[:, comb], y_idx, len(uniq))})
    df = pd.DataFrame(filas).sort_values("Separation (1-Wilks)", ascending=False).reset_index(drop=True)
    df["CV balanced accuracy"] = np.nan
    cuenta = np.bincount(y_idx)
    cv = int(min(5, cuenta.min()))
    if cv >= 2:
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=0)
        for i in range(min(n_confirmar, len(df))):
            S = scores[:, list(df.loc[i, "cols"])]
            try:
                pred = cross_val_predict(LinearDiscriminantAnalysis(), S, y_idx, cv=skf)
                df.loc[i, "CV balanced accuracy"] = balanced_accuracy_score(y_idx, pred)
            except Exception:
                pass
    df["PCs"] = df["cols"].apply(lambda c: " + ".join(f"PC{j + 1}" for j in c))
    return df


def buscar_regresion(scores, y, k_max, tam, n_confirmar=12):
    y = np.asarray(y, dtype=float)
    n = len(y)
    k_max = min(k_max, scores.shape[1])
    ss_tot = float(((y - y.mean()) ** 2).sum()) or 1.0
    filas = []
    for comb in combinations(range(k_max), tam):
        S = scores[:, comb]
        A = np.column_stack([np.ones(n), S])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        r2 = 1.0 - float(((y - A @ coef) ** 2).sum()) / ss_tot
        adj = 1.0 - (1.0 - r2) * (n - 1) / max(n - tam - 1, 1)
        filas.append({"cols": comb, "R² (fit)": r2, "Adjusted R²": adj})
    df = pd.DataFrame(filas).sort_values("Adjusted R²", ascending=False).reset_index(drop=True)
    df["Q² (CV)"] = np.nan
    cv = int(min(5, max(2, n // 4)))
    kf = KFold(n_splits=cv, shuffle=True, random_state=0)
    for i in range(min(n_confirmar, len(df))):
        S = scores[:, list(df.loc[i, "cols"])]
        try:
            pred = cross_val_predict(LinearRegression(), S, y, cv=kf)
            df.loc[i, "Q² (CV)"] = 1.0 - float(((y - pred) ** 2).sum()) / ss_tot
        except Exception:
            pass
    df["PCs"] = df["cols"].apply(lambda c: " + ".join(f"PC{j + 1}" for j in c))
    return df


def direccion_variables(scores, cargas, cols, clases=None, y=None):
    """Importance of every original variable for the chosen PC combination.
    Returns (weights, signed): `weights` >= 0 (overall importance, 0-1 scaled); `signed` is the
    signed curve when it is meaningful (2 classes, or regression), otherwise None."""
    cols = list(cols)
    V = cargas[cols, :]                       # (d, p)
    S = scores[:, cols]
    if y is not None:
        A = np.column_stack([np.ones(len(y)), S])
        coef, *_ = np.linalg.lstsq(A, np.asarray(y, dtype=float), rcond=None)
        w = coef[1:] @ V                      # (p,) direction of increasing concentration
        signed = w
        imp = np.abs(w)
    else:
        lda = LinearDiscriminantAnalysis(solver="svd").fit(S, np.asarray(clases).astype(str))
        W = lda.scalings_[:, :max(1, len(lda.classes_) - 1)]          # (d, c-1)
        Wv = V.T @ W                                                  # (p, c-1)
        ev = np.asarray(lda.explained_variance_ratio_)[:Wv.shape[1]]
        imp = np.sqrt((Wv ** 2 * ev).sum(axis=1))
        signed = Wv[:, 0] if len(lda.classes_) == 2 else None
    m = imp.max() or 1.0
    return imp / m, (None if signed is None else signed / (np.abs(signed).max() or 1.0))


def regiones_importantes(eje, imp, percentil=95, ancho_suavizado=3, max_regiones=15):
    """Contiguous stretches of the axis whose importance is in the top percentile."""
    eje = np.asarray(eje, dtype=float)
    s = uniform_filter1d(imp, size=max(1, int(ancho_suavizado)), mode="nearest")
    umbral = np.percentile(s, percentil)
    lab, n = label(s >= umbral)
    filas = []
    for k in range(1, n + 1):
        idx = np.where(lab == k)[0]
        j = idx[np.argmax(s[idx])]
        filas.append({"From": float(eje[idx[0]]), "To": float(eje[idx[-1]]), "Peak at": float(eje[j]),
                      "Importance (max)": float(s[j]), "Variables": int(len(idx))})
    df = pd.DataFrame(filas)
    if len(df):
        df = df.sort_values("Importance (max)", ascending=False).head(max_regiones).reset_index(drop=True)
    return df, float(umbral)


def picos_importantes(eje, imp, signed=None, n=15):
    eje = np.asarray(eje, dtype=float)
    s = uniform_filter1d(imp, size=3, mode="nearest")
    pk, _ = find_peaks(s, prominence=0.05 * (s.max() or 1.0))
    if len(pk) == 0:
        pk = np.array([int(np.argmax(s))])
    pk = pk[np.argsort(s[pk])[::-1]][:n]
    filas = [{"Position": float(eje[j]), "Importance": float(s[j]),
              **({"Direction": ("+" if signed[j] > 0 else "−")} if signed is not None else {})} for j in pk]
    return pd.DataFrame(filas)
