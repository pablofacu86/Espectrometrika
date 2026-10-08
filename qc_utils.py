"""Quality-control charts for instrument/method monitoring: multivariate (T2, Q, PC scores, EWMA),
acquisition metrics of a stable sample and Levey-Jennings charts with Westgard rules."""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import savgol_filter


# ------------------------------------------------------------------ pre-processing of QC spectra
def preparar(X, modo="none", ventana=11):
    X = np.asarray(X, dtype=float)
    if modo == "snv":
        s = X.std(axis=1, keepdims=True)
        return (X - X.mean(axis=1, keepdims=True)) / np.where(s > 0, s, 1)
    if modo == "d1":
        return savgol_filter(X, int(ventana) | 1, 2, deriv=1, axis=1)
    if modo == "d2":
        return savgol_filter(X, int(ventana) | 1, 3, deriv=2, axis=1)
    return X


# ------------------------------------------------------------------ multivariate model
def ajustar_pca_qc(Xb, k=None, var=0.95, k_max=8):
    Xb = np.asarray(Xb, dtype=float)
    n, p = Xb.shape
    mu = Xb.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xb - mu, full_matrices=False)
    lam_all = S ** 2 / max(n - 1, 1)
    if k is None:
        cum = np.cumsum(lam_all) / max(lam_all.sum(), 1e-12)
        k = int(np.searchsorted(cum, var) + 1)
    k = int(max(1, min(k, k_max, n - 2, len(S))))
    V = Vt[:k]
    sc = (Xb - mu) @ V.T
    t2 = (sc ** 2 / lam_all[:k]).sum(axis=1)
    q = ((Xb - mu - sc @ V.T @ V) ** 2).sum(axis=1) if False else ((Xb - mu - sc @ V) ** 2).sum(axis=1)

    def nu(v):
        m, sd = float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
        return max(m, 1e-12), int(np.clip(round(2 * (m / sd) ** 2) if sd > 0 else 1, 1, 200))
    # In-sample T2/Q are optimistic when there are few baseline runs and many variables (the components fit the noise of the
    # baseline itself). Limits are therefore estimated from LEAVE-ONE-OUT statistics: each baseline run is scored by a model
    # that has not seen it, exactly like a future run.
    t2_loo, q_loo = np.empty(n), np.empty(n)
    for i in range(n):
        idx = np.arange(n) != i
        mi = Xb[idx].mean(axis=0)
        Ui, Si, Vti = np.linalg.svd(Xb[idx] - mi, full_matrices=False)
        ki = int(max(1, min(k, len(Si))))
        lam_i = Si[:ki] ** 2 / max(n - 2, 1)
        sci = (Xb[i] - mi) @ Vti[:ki].T
        t2_loo[i] = float((sci ** 2 / lam_i).sum())
        q_loo[i] = float(((Xb[i] - mi - sci @ Vti[:ki]) ** 2).sum())
    h0, Nh = nu(t2_loo)
    q0, Nq = nu(q_loo)
    return {"mu": mu, "V": V, "lam": lam_all[:k], "k": k, "h0": h0, "Nh": Nh, "q0": q0, "Nq": Nq,
            "sc_mu": sc.mean(axis=0), "sc_sd": sc.std(axis=0, ddof=1) if n > 1 else np.ones(k), "n": n,
            "t2_loo": t2_loo, "q_loo": q_loo}


def evaluar_qc(m, X, alpha_w=0.05, alpha_a=0.0027):
    X = np.asarray(X, dtype=float)
    sc = (X - m["mu"]) @ m["V"].T
    t2 = (sc ** 2 / m["lam"]).sum(axis=1)
    q = ((X - m["mu"] - sc @ m["V"]) ** 2).sum(axis=1)
    if "t2_loo" in m and len(X) >= m["n"]:           # the first n rows are the baseline: use their out-of-sample values
        t2 = t2.copy(); q = q.copy()
        t2[:m["n"]] = m["t2_loo"]; q[:m["n"]] = m["q_loo"]
    lim = lambda h0, N, a: h0 * stats.chi2.ppf(1 - a, N) / N
    lims = {"T2_warn": lim(m["h0"], m["Nh"], alpha_w), "T2_act": lim(m["h0"], m["Nh"], alpha_a),
            "Q_warn": lim(m["q0"], m["Nq"], alpha_w), "Q_act": lim(m["q0"], m["Nq"], alpha_a)}
    return sc, t2, q, lims


# ------------------------------------------------------------------ univariate charts
def ewma(x, lam=0.2, mu0=None, sd0=None, L=3.0):
    x = np.asarray(x, dtype=float)
    mu0 = float(np.mean(x)) if mu0 is None else mu0
    sd0 = float(np.std(x, ddof=1)) if sd0 is None and len(x) > 1 else (sd0 or 1.0)
    z = np.empty(len(x)); prev = mu0
    for i, v in enumerate(x):
        prev = lam * v + (1 - lam) * prev
        z[i] = prev
    i = np.arange(1, len(x) + 1)
    w = L * sd0 * np.sqrt(lam / (2 - lam) * (1 - (1 - lam) ** (2 * i)))
    return z, mu0 - w, mu0 + w


def westgard(vals, mean, sd):
    """Westgard multi-rule flags per point: 1-3s, 2-2s, R-4s, 4-1s, 10-x."""
    z = (np.asarray(vals, dtype=float) - mean) / (sd if sd > 0 else 1.0)
    flags = [[] for _ in z]
    for i, zi in enumerate(z):
        if abs(zi) > 3:
            flags[i].append("1-3s")
        if i >= 1 and ((z[i] > 2 and z[i - 1] > 2) or (z[i] < -2 and z[i - 1] < -2)):
            flags[i].append("2-2s")
        if i >= 1 and abs(z[i] - z[i - 1]) > 4:
            flags[i].append("R-4s")
        if i >= 3 and (all(z[i - 3:i + 1] > 1) or all(z[i - 3:i + 1] < -1)):
            flags[i].append("4-1s")
        if i >= 9 and (all(z[i - 9:i + 1] > 0) or all(z[i - 9:i + 1] < 0)):
            flags[i].append("10-x")
    return z, [", ".join(f) for f in flags]


def metricas_adquisicion(X, eje, pos, semiancho):
    """Per-spectrum metrics: height and position of the peak in [pos±semiancho], area, baseline level, noise, SNR."""
    X = np.asarray(X, dtype=float); eje = np.asarray(eje, dtype=float)
    m = (eje >= pos - semiancho) & (eje <= pos + semiancho)
    if m.sum() < 3:
        raise ValueError("The peak window has fewer than 3 points — widen it.")
    xe = eje[m]
    filas = []
    for s in X:
        seg = s[m]
        base_local = np.percentile(seg, 5)
        alt = seg - base_local
        j = int(np.argmax(seg))
        cen = float(np.sum(xe * alt) / np.sum(alt)) if np.sum(alt) > 0 else float(xe[j])
        ruido = float(np.std(s - savgol_filter(s, 11, 2)))
        filas.append({"peak_height": float(seg[j] - base_local), "peak_position": float(xe[j]),
                      "peak_centroid": cen, "peak_area": float(np.trapz(alt, xe)) if hasattr(np, "trapz") else float(np.trapezoid(alt, xe)),
                      "baseline_level": float(np.percentile(s, 5)), "noise": ruido,
                      "SNR": float((seg[j] - base_local) / ruido) if ruido > 0 else np.nan})
    return pd.DataFrame(filas)
