"""Calibration transfer between instruments: offset, direct standardization (DS) and piecewise DS (PDS)."""
import numpy as np


def _trunc_inv(A, rcond=1e-3, rango=None):
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    k = int((S > rcond * S[0]).sum())
    if rango:
        k = min(k, int(rango))
    return U[:, :k], S[:k], Vt[:k]


def ajustar(Xm, Xs, metodo="PDS", ventana=5, rango=None):
    """Xm: master spectra of the transfer samples; Xs: the same samples on the secondary instrument (same axis)."""
    Xm = np.asarray(Xm, dtype=float); Xs = np.asarray(Xs, dtype=float)
    mm, ms = Xm.mean(axis=0), Xs.mean(axis=0)
    if metodo == "Offset":
        return {"metodo": metodo, "mm": mm, "ms": ms, "d": mm - ms}
    Xmc, Xsc = Xm - mm, Xs - ms
    if metodo == "DS":
        U, S, Vt = _trunc_inv(Xsc, rango=rango)
        A = Vt.T                                       # p x k
        B = (U.T @ Xmc) / S[:, None]                   # k x p
        return {"metodo": metodo, "mm": mm, "ms": ms, "A": A, "B": B}
    # PDS: every master variable j is modelled from a window of secondary variables around j
    p = Xs.shape[1]
    w = int(ventana)
    coefs, lims = [], []
    for j in range(p):
        a, b = max(0, j - w), min(p, j + w + 1)
        U, S, Vt = _trunc_inv(Xsc[:, a:b], rango=rango)
        beta = Vt.T @ ((U.T @ Xmc[:, j]) / S)
        coefs.append(beta); lims.append((a, b))
    return {"metodo": metodo, "mm": mm, "ms": ms, "coefs": coefs, "lims": lims}


def aplicar(mod, Xnew):
    Xnew = np.asarray(Xnew, dtype=float)
    if mod["metodo"] == "Offset":
        return Xnew + mod["d"]
    Xc = Xnew - mod["ms"]
    if mod["metodo"] == "DS":
        return Xc @ mod["A"] @ mod["B"] + mod["mm"]
    out = np.empty_like(Xnew)
    for j, (beta, (a, b)) in enumerate(zip(mod["coefs"], mod["lims"])):
        out[:, j] = Xc[:, a:b] @ beta
    return out + mod["mm"]


def evaluar_cv(Xm, Xs, metodo, ventana, rango, k=None):
    """Leave-out evaluation on the transfer samples: RMS difference to the master before and after the correction."""
    n = len(Xm)
    k = n if (k is None or k >= n) else k
    idx = np.arange(n)
    pasos = np.array_split(idx, k)
    corr = np.empty_like(Xs)
    for te in pasos:
        tr = np.setdiff1d(idx, te)
        mod = ajustar(Xm[tr], Xs[tr], metodo, ventana, rango)
        corr[te] = aplicar(mod, Xs[te])
    rms = lambda A: float(np.sqrt(np.mean((A - Xm) ** 2)))
    return rms(Xs), rms(corr), corr
