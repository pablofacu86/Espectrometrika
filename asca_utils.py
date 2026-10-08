"""ASCA (ANOVA-simultaneous component analysis) for 1 or 2 factors (+ interaction).
Sequential (type-I) effects through orthonormal bases, permutation p-values, PCA of each effect."""
import numpy as np


def _dummy(lab):
    lab = np.asarray(lab).astype(str)
    niv = list(dict.fromkeys(lab))
    if len(niv) < 2:
        raise ValueError("a factor needs at least 2 levels")
    D = np.zeros((len(lab), len(niv) - 1))
    for j, v in enumerate(niv[:-1]):
        D[lab == v, j] = 1.0
    D[lab == niv[-1], :] = -1.0          # sum-to-zero coding
    return D, niv


def _orth(D, previas, tol=1e-9):
    D = D - D.mean(0)
    for Q in previas:
        D = D - Q @ (Q.T @ D)
    if D.size == 0:
        return np.zeros((D.shape[0], 0))
    U, S, _ = np.linalg.svd(D, full_matrices=False)
    k = int((S > tol * max(S.max(), 1e-30)).sum()) if S.size else 0
    return U[:, :k]


def asca(X, factores, interaccion=True, n_perm=500, n_comp=2, semilla=0):
    """factores: dict {name: labels} with 1 or 2 entries. Returns a dict with a summary table and the PCA of each effect."""
    import pandas as pd
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    nombres = list(factores)
    if not 1 <= len(nombres) <= 2:
        raise ValueError("use 1 or 2 factors")
    Xc = X - X.mean(0)
    Ds = {nm: _dummy(factores[nm]) for nm in nombres}
    bases, orden, previas = {}, [], []
    for nm in nombres:
        Q = _orth(Ds[nm][0], previas); bases[nm] = Q; previas.append(Q); orden.append(nm)
    if len(nombres) == 2 and interaccion:
        a, b = nombres
        DI = np.hstack([Ds[a][0][:, [i]] * Ds[b][0][:, [j]] for i in range(Ds[a][0].shape[1]) for j in range(Ds[b][0].shape[1])])
        nmi = f"{a} × {b}"
        bases[nmi] = _orth(DI, previas); orden.append(nmi)
    tot = float((Xc ** 2).sum())
    ss = {nm: float(((bases[nm].T @ Xc) ** 2).sum()) for nm in orden}
    rng = np.random.default_rng(semilla)
    cuenta = {nm: 0 for nm in orden}
    for _ in range(int(n_perm)):
        Xp = Xc[rng.permutation(n)]
        for nm in orden:
            if bases[nm].shape[1] and float(((bases[nm].T @ Xp) ** 2).sum()) >= ss[nm]:
                cuenta[nm] += 1
    efectos = {nm: bases[nm] @ (bases[nm].T @ Xc) for nm in orden}
    resid = Xc - sum(efectos.values())
    ss_res = float((resid ** 2).sum())
    filas = [{"Effect": nm, "d.f.": int(bases[nm].shape[1]), "Sum of squares": ss[nm], "% of total variance": round(100 * ss[nm] / tot, 2) if tot > 0 else 0.0,
              "p-value (permutation)": round((1 + cuenta[nm]) / (1 + n_perm), 4) if bases[nm].shape[1] else np.nan} for nm in orden]
    filas.append({"Effect": "Residual", "d.f.": int(n - 1 - sum(bases[nm].shape[1] for nm in orden)), "Sum of squares": ss_res,
                  "% of total variance": round(100 * ss_res / tot, 2) if tot > 0 else 0.0, "p-value (permutation)": np.nan})
    pcs = {}
    for nm in orden:
        M = efectos[nm]
        if bases[nm].shape[1] == 0 or ss[nm] <= 0:
            continue
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        k = int(min(n_comp, (S > 1e-9 * S[0]).sum()))
        V = Vt[:k].T
        pcs[nm] = {"loadings": V, "scores": M @ V, "scores_with_residual": (M + resid) @ V,
                   "var": (S[:k] ** 2) / (S ** 2).sum()}
    return {"table": pd.DataFrame(filas), "pcs": pcs, "levels": {nm: Ds[nm][1] for nm in nombres}}
