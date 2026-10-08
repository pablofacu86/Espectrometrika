"""Light hyperspectral-image tools: cube loading, spatial/spectral reduction, PCA score maps, k-means class maps."""
import io
import numpy as np
import pandas as pd

MAX_VALORES = 25_000_000      # pixels x bands kept in memory after reduction


def cargar_cubo(archivo):
    """.npy (H,W,B) | .npz (cube [+ axis]) | long table .csv/.xlsx with columns x,y followed by the spectral columns.
    Returns (cube float32 H x W x B, axis float array)."""
    nom = archivo.name.lower()
    if nom.endswith(".npy"):
        cub = np.load(archivo, allow_pickle=False)
        eje = np.arange(cub.shape[2], dtype=float)
    elif nom.endswith(".npz"):
        z = np.load(archivo, allow_pickle=False)
        clave = "cube" if "cube" in z.files else z.files[0]
        cub = z[clave]
        eje = np.asarray(z["axis"], dtype=float) if "axis" in z.files else np.arange(cub.shape[2], dtype=float)
    else:
        df = pd.read_csv(archivo) if nom.endswith(".csv") else pd.read_excel(archivo)
        cols = [str(c).strip().lower() for c in df.columns]
        if len(cols) < 4 or cols[0] not in ("x", "col") or cols[1] not in ("y", "row"):
            raise ValueError("a table needs the first two columns named x and y, then one column per band")
        xs, ys = df.iloc[:, 0].astype(int).to_numpy(), df.iloc[:, 1].astype(int).to_numpy()
        xs -= xs.min(); ys -= ys.min()
        eje = np.array([float(str(c).replace(",", ".")) for c in df.columns[2:]])
        cub = np.full((ys.max() + 1, xs.max() + 1, len(eje)), np.nan, dtype=np.float32)
        cub[ys, xs, :] = df.iloc[:, 2:].to_numpy(dtype=np.float32)
    if cub.ndim != 3:
        raise ValueError("the cube must have 3 dimensions (rows, columns, bands)")
    if len(eje) != cub.shape[2]:
        eje = np.arange(cub.shape[2], dtype=float)
    return cub.astype(np.float32), np.asarray(eje, dtype=float)


def reducir(cub, eje, paso_esp=1, paso_spec=1):
    """Keep every paso_esp-th pixel in both directions (mean of the block) and every paso_spec-th band (mean of the block)."""
    H, W, B = cub.shape
    if paso_esp > 1:
        h, w = H // paso_esp * paso_esp, W // paso_esp * paso_esp
        cub = np.nanmean(cub[:h, :w].reshape(h // paso_esp, paso_esp, w // paso_esp, paso_esp, B), axis=(1, 3)).astype(np.float32)
    if paso_spec > 1:
        b = B // paso_spec * paso_spec
        cub = cub[:, :, :b].reshape(cub.shape[0], cub.shape[1], b // paso_spec, paso_spec).mean(axis=3)
        eje = eje[:b].reshape(-1, paso_spec).mean(axis=1)
    return cub, eje


def auto_pasos(shape):
    H, W, B = shape
    pe = 1
    while (H // pe) * (W // pe) * B > MAX_VALORES:
        pe += 1
    return pe


def matriz_pixeles(cub, pretrat="none", umbral_pct=0.0):
    """Pixels x bands matrix of the valid pixels + the boolean mask of the image (True = analysed)."""
    H, W, B = cub.shape
    M = cub.reshape(-1, B).astype(np.float64)
    ok = np.isfinite(M).all(axis=1)
    if umbral_pct > 0:                       # drop the darkest pixels (background): mean intensity below the percentile
        inten = np.where(ok, np.nanmean(M, axis=1), np.nan)
        ok &= inten >= np.nanpercentile(inten[ok], umbral_pct)
    M = M[ok]
    if pretrat == "snv":
        sd = M.std(axis=1, keepdims=True); M = (M - M.mean(axis=1, keepdims=True)) / np.where(sd > 0, sd, 1)
    elif pretrat == "norm":
        nr = np.linalg.norm(M, axis=1, keepdims=True); M = M / np.where(nr > 0, nr, 1)
    return M, ok.reshape(H, W)


def pca_mapas(M, ok, n_comp=3):
    mu = M.mean(0); Xc = M - mu
    # economical: randomised SVD on a subsample of pixels for the loadings, then project every pixel
    rng = np.random.default_rng(0)
    idx = rng.choice(len(Xc), size=min(len(Xc), 20000), replace=False)
    U, S, Vt = np.linalg.svd(Xc[idx], full_matrices=False)
    k = int(min(n_comp, len(S)))
    V = Vt[:k].T
    T = Xc @ V
    var = (S[:k] ** 2) / (S ** 2).sum()
    imgs = []
    for j in range(k):
        im = np.full(ok.shape, np.nan); im[ok] = T[:, j]; imgs.append(im)
    return imgs, V, var


def kmeans_mapa(M, ok, k=4, semilla=0):
    from sklearn.cluster import MiniBatchKMeans
    km = MiniBatchKMeans(n_clusters=int(k), random_state=semilla, n_init=3, batch_size=2048).fit(M)
    lab = km.labels_
    im = np.full(ok.shape, np.nan); im[ok] = lab
    medias = np.array([M[lab == c].mean(axis=0) for c in range(int(k))])
    return im, medias, np.bincount(lab, minlength=int(k))


def cubo_demo(h=40, w=48, semilla=0):
    """Simulated Raman-like image of a tablet: matrix + 2 particles types + background."""
    rng = np.random.default_rng(semilla)
    x = np.arange(400, 1801, 7.0)
    L = lambda c, a, hh: hh / (1 + ((x - c) / a) ** 2)
    sp = {"paracetamol": L(1648, 8, 1) + L(1324, 7, .8) + L(858, 6, .7) + L(797, 6, .4),
          "lactose": L(1126, 8, .8) + L(1090, 8, .7) + L(876, 8, .5) + L(520, 9, .5),
          "ibuprofen": L(1608, 7, .7) + L(1183, 7, .4) + L(1002, 6, .5) + L(745, 6, .5)}
    cub = np.zeros((h, w, len(x)), dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    comp = np.zeros((3, h, w))
    comp[1] = 0.8
    for _ in range(14):
        cy, cx, r = rng.integers(3, h - 3), rng.integers(3, w - 3), rng.uniform(2, 5)
        m = (yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2
        t = rng.integers(0, 3)
        comp[:, m] = 0; comp[t, m] = 1
    names = list(sp)
    for i in range(h):
        for j in range(w):
            s = sum(comp[k, i, j] * sp[names[k]] for k in range(3))
            cub[i, j] = s + 0.3 * np.exp(-((x - 1000) / 700) ** 2) + rng.normal(0, 0.03, len(x))
    return cub, x
