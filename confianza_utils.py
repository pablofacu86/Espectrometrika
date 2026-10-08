"""Reliability tools for predictions: conformal prediction, applicability domain (DD-SIMCA style
distances) and per-sample explanation by occlusion of spectral windows."""
import numpy as np
import pandas as pd
from scipy import stats


# ----------------------------------------------------------------------------- conformal
def _cuantil_conforme(scores, alpha):
    s = np.sort(np.asarray(scores, dtype=float))
    n = len(s)
    if n == 0:
        return np.inf
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.inf if k > n else float(s[k - 1])


def calibracion_desde_cv(cv, es_clf):
    """Calibration data from the out-of-fold cross-validation predictions of a trained model."""
    if es_clf:
        if cv.get("y_proba") is None:
            return None
        return {"tipo": "clf", "proba": np.asarray(cv["y_proba"], dtype=float), "y": np.asarray(cv["y_true"]),
                "clases": np.asarray(cv["clases_proba"])}
    return {"tipo": "reg", "y": np.asarray(cv["y_true"], dtype=float), "yhat": np.asarray(cv["y_pred"], dtype=float)}


def intervalos_regresion(cal, yhat, alpha=0.1, por_rango=False):
    """Split-conformal (out-of-fold residuals). por_rango=True: separate quantile for the lower/middle/upper
    third of the predicted values (handles error that grows with concentration)."""
    yhat = np.asarray(yhat, dtype=float).ravel()
    res = np.abs(cal["y"] - cal["yhat"])
    if not por_rango or len(res) < 30:
        q = np.full(len(yhat), _cuantil_conforme(res, alpha))
        return yhat - q, yhat + q, q
    cortes = np.quantile(cal["yhat"], [1 / 3, 2 / 3])
    cb = np.digitize(cal["yhat"], cortes)
    qs = [_cuantil_conforme(res[cb == b], alpha) for b in range(3)]
    qn = np.digitize(yhat, cortes)
    q = np.array([qs[b] for b in qn])
    return yhat - q, yhat + q, q


def conjuntos_clasificacion(cal, proba, clases_modelo, alpha=0.1):
    """Class-conditional conformal prediction sets (one threshold per class, so rare classes keep their coverage).
    Returns (list of sets as text, set sizes)."""
    clases_cal = list(cal["clases"])
    P = np.asarray(proba, dtype=float)
    umbrales = {}
    for k, c in enumerate(clases_cal):
        m = cal["y"] == c
        if m.sum() == 0:
            umbrales[c] = np.inf
            continue
        j = clases_cal.index(c)
        umbrales[c] = _cuantil_conforme(1 - cal["proba"][m, j], alpha)
    sets, tam = [], []
    for i in range(P.shape[0]):
        incl = [c for c in clases_modelo if (1 - P[i, list(clases_modelo).index(c)]) <= umbrales.get(c, np.inf)]
        sets.append(" | ".join(map(str, incl)) if incl else "∅ (none — unusual sample)")
        tam.append(len(incl))
    return sets, np.array(tam)


# ----------------------------------------------------------------------------- applicability domain
def ajustar_dominio(X, var_pca=0.95, k_max=15):
    """PCA model of the training samples + DD-SIMCA scaling of the score (h) and orthogonal (q) distances."""
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    mu = X.mean(axis=0)
    Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    lam_all = S ** 2 / max(n - 1, 1)
    cum = np.cumsum(lam_all) / max(lam_all.sum(), 1e-12)
    k = int(np.searchsorted(cum, var_pca) + 1)
    k = int(max(2, min(k, k_max, n - 2, len(S))))
    V = Vt[:k]
    lam = lam_all[:k]
    sc = Xc @ V.T
    h = (sc ** 2 / lam).sum(axis=1)
    q = ((Xc - sc @ V) ** 2).sum(axis=1)

    def _nu(v):
        m, s = float(np.mean(v)), float(np.std(v, ddof=1))
        return m, int(np.clip(round(2 * (m / s) ** 2) if s > 0 else 1, 1, 100))
    h0, Nh = _nu(h)
    q0, Nq = _nu(q)
    return {"mu": mu, "V": V, "lam": lam, "k": k, "h0": h0, "Nh": Nh, "q0": max(q0, 1e-12), "Nq": Nq,
            "h_train": h / h0, "q_train": q / max(q0, 1e-12), "n": n, "var_expl": float(cum[k - 1])}


def distancias_dominio(dom, X, alpha=0.05):
    Xc = np.asarray(X, dtype=float) - dom["mu"]
    sc = Xc @ dom["V"].T
    h = (sc ** 2 / dom["lam"]).sum(axis=1) / dom["h0"]
    q = ((Xc - sc @ dom["V"]) ** 2).sum(axis=1) / dom["q0"]
    c = dom["Nh"] * h + dom["Nq"] * q
    df = dom["Nh"] + dom["Nq"]
    crit = float(stats.chi2.ppf(1 - alpha, df))
    crit_ext = float(stats.chi2.ppf((1 - alpha) ** (1 / max(dom["n"], 1)), df))
    estado = np.where(c <= crit, "inside", np.where(c <= crit_ext, "outside (borderline)", "outside (extreme)"))
    return pd.DataFrame({"h (score distance)": h, "q (residual distance)": q, "total distance": c,
                         "domain": estado}), crit


def curva_aceptacion(dom, crit, x_max):
    x = np.linspace(0, min(x_max, crit / dom["Nh"]), 200)
    y = (crit - dom["Nh"] * x) / dom["Nq"]
    return x, np.clip(y, 0, None)


# ----------------------------------------------------------------------------- explanation
def explicar_por_ventanas(modelo, x, ref, es_clf, n_ventanas=40, clase=None):
    """Occlusion: replace each window of the sample by the reference (training mean) and see how the
    prediction changes. Returns (windows, effect) — positive effect = the window pushes the prediction UP
    (probability of `clase` / predicted value)."""
    x = np.asarray(x, dtype=float).ravel()
    ref = np.asarray(ref, dtype=float).ravel()
    p = len(x)
    nw = int(max(2, min(n_ventanas, p)))
    cortes = np.linspace(0, p, nw + 1).astype(int)
    wins = [np.arange(cortes[i], cortes[i + 1]) for i in range(nw) if cortes[i + 1] > cortes[i]]
    M = np.tile(x, (len(wins) + 1, 1))
    for i, w in enumerate(wins):
        M[i + 1, w] = ref[w]
    if es_clf:
        P = modelo.predict_proba(M)
        j = list(modelo.classes_).index(clase)
        s = P[:, j]
    else:
        s = np.asarray(modelo.predict(M), dtype=float).ravel()
    return wins, s[0] - s[1:], float(s[0])
