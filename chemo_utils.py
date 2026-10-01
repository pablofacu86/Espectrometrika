"""
chemo_utils.py
Funciones de quimiometría reutilizadas por la app interactiva
(la misma lógica ya validada en el notebook de la clase práctica).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.cluster import hierarchy
from sklearn.decomposition import PCA
from scipy.stats import f as f_dist
from scipy.stats import norm


# =============================================================================
# PRETRATAMIENTOS
# =============================================================================

def normalizar(X, modo="area"):
    """Normalización por fila (por muestra). modo: 'area', 'vector' o 'minmax'."""
    X = np.asarray(X, dtype=float)
    if modo == "area":
        denom = np.sum(np.abs(X), axis=1, keepdims=True)
    elif modo == "vector":
        denom = np.linalg.norm(X, axis=1, keepdims=True)
    elif modo == "minmax":
        minimo = X.min(axis=1, keepdims=True)
        maximo = X.max(axis=1, keepdims=True)
        rango = maximo - minimo
        rango[rango == 0] = 1
        return (X - minimo) / rango
    else:
        raise ValueError("modo debe ser 'area', 'vector' o 'minmax'")
    denom[denom == 0] = 1
    return X / denom


def estandarizar(X):
    """Autoescalado por columna (por número de onda): media 0, desvío 1 entre muestras."""
    X = np.asarray(X, dtype=float)
    media = X.mean(axis=0, keepdims=True)
    desvio = X.std(axis=0, keepdims=True)
    desvio[desvio == 0] = 1
    return (X - media) / desvio


def snv(X):
    """Standard Normal Variate, por fila (por muestra)."""
    X = np.asarray(X, dtype=float)
    media = X.mean(axis=1, keepdims=True)
    desvio = X.std(axis=1, keepdims=True)
    desvio[desvio == 0] = 1
    return (X - media) / desvio


def msc(X, referencia=None):
    """Multiplicative Scatter Correction."""
    X = np.asarray(X, dtype=float)
    if referencia is None:
        referencia = X.mean(axis=0)
    X_msc = np.zeros_like(X)
    for i in range(X.shape[0]):
        b, a = np.polyfit(referencia, X[i, :], 1)
        if b == 0:
            b = 1e-12
        X_msc[i, :] = (X[i, :] - a) / b
    return X_msc


def suavizado_sg(X, ventana=11, orden_polinomio=2):
    """Suavizado de Savitzky-Golay (sin derivar)."""
    return savgol_filter(X, window_length=ventana, polyorder=orden_polinomio, deriv=0, axis=1)


def derivada_sg(X, orden, ventana=11, orden_polinomio=2):
    """Derivada de Savitzky-Golay (1ra o 2da)."""
    return savgol_filter(X, window_length=ventana, polyorder=orden_polinomio, deriv=orden, axis=1)


def aplicar_pretratamientos(X, pasos):
    """
    Aplica una secuencia de pretratamientos en el orden dado.
    'pasos' es una lista de tuplas (nombre, parametros_dict).
    """
    X_out = np.asarray(X, dtype=float).copy()
    funciones = {
        "normalizar": normalizar,
        "estandarizar": estandarizar,
        "snv": snv,
        "msc": msc,
        "suavizado_sg": suavizado_sg,
        "derivada_sg": derivada_sg,
    }
    for nombre, params in pasos:
        X_out = funciones[nombre](X_out, **params)
    return X_out


# =============================================================================
# DETECCIÓN DE OUTLIERS: T2 DE HOTELLING Y RESIDUAL Q
# =============================================================================

def calcular_T2(scores, autovalores, n_comp):
    return np.sum((scores[:, :n_comp] ** 2) / autovalores[:n_comp], axis=1)


def limite_T2(n_muestras, n_comp, alpha=0.05):
    F_critico = f_dist.ppf(1 - alpha, n_comp, n_muestras - n_comp)
    return n_comp * (n_muestras - 1) / (n_muestras - n_comp) * F_critico


def calcular_Q(X, scores, cargas, n_comp, media=None):
    """
    Q residual (SPE): la parte de la variabilidad de cada muestra que el
    modelo (con 'n_comp' componentes) NO logra explicar.
    'media' debe ser la media usada para centrar los datos antes del PCA
    (pca.mean_). Si no se pasa, se asume que X ya llega centrado — pero
    normalmente NO es el caso (los pretratamientos como SNV/MSC centran
    cada espectro por fila, no cada variable por columna), así que hay
    que pasarla siempre que se tenga disponible.
    """
    X_centrado = X - media if media is not None else X
    X_reconstruido = scores[:, :n_comp] @ cargas[:n_comp, :]
    residuos = X_centrado - X_reconstruido
    return np.sum(residuos ** 2, axis=1)


def limite_Q(autovalores, n_comp, alpha=0.05):
    """Aproximación de Jackson-Mudholkar. Devuelve np.inf si no es calculable."""
    autov_resto = np.clip(autovalores[n_comp:], 0, None)
    theta1 = np.sum(autov_resto)
    varianza_total = np.sum(np.clip(autovalores, 0, None))

    if theta1 < 1e-8 * varianza_total:
        return np.inf

    theta2 = np.sum(autov_resto ** 2)
    theta3 = np.sum(autov_resto ** 3)
    h0 = 1 - (2 * theta1 * theta3) / (3 * theta2 ** 2)
    z_alpha = norm.ppf(1 - alpha)
    termino = (z_alpha * np.sqrt(2 * theta2 * h0 ** 2) / theta1
               + 1 + theta2 * h0 * (h0 - 1) / theta1 ** 2)
    return theta1 * termino ** (1 / h0)


def limite_Q_confiable(Q, Q_lim):
    """True si el límite teórico de Q es razonable frente a los datos observados."""
    return np.isfinite(Q_lim) and Q_lim >= np.percentile(Q, 5)


def rango_con_margen(valores, limite, margen=0.08):
    """
    Rango [lo, hi] para un eje, haciendo zoom a donde están los datos en vez
    de forzar siempre a arrancar en 0 (T² y Q rara vez valen 0 en la práctica).
    """
    lo = min(np.min(valores), limite)
    hi = max(np.max(valores), limite)
    ancho = hi - lo
    if ancho <= 0:
        ancho = hi if hi > 0 else 1.0
    lo = max(0, lo - margen * ancho)
    hi = hi + margen * ancho
    return lo, hi


# =============================================================================
# DENDROGRAMA CIRCULAR
# =============================================================================

def dendrograma_circular_fig(Z, etiquetas, figsize=(8, 8), umbral_color=None,
                              id_a_clase=None, mapa_color_clase=None):
    """
    Returns a matplotlib figure with the dendrogram in polar projection.
    If id_a_clase (dict: sample id -> class) and mapa_color_clase (dict:
    class -> hex color) are given, each leaf label is colored by its class,
    so cluster purity (whether same-class samples group together) is easy
    to see at a glance.
    """
    dend = hierarchy.dendrogram(Z, labels=etiquetas, no_plot=True, color_threshold=umbral_color)
    icoord = np.array(dend["icoord"])
    dcoord = np.array(dend["dcoord"])
    ivl = dend["ivl"]
    colores = dend["color_list"]

    n_hojas = len(ivl)
    x_max = icoord.max()
    theta = icoord / x_max * (2 * np.pi)
    d_max = dcoord.max()
    r = d_max - dcoord

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="polar")
    for xs, ys, c in zip(theta, r, colores):
        ax.plot(xs, ys, color=c, linewidth=1.2)

    pos_hojas = (5 + 10 * np.arange(n_hojas)) / x_max * (2 * np.pi)
    for ang, nombre in zip(pos_hojas, ivl):
        rot = np.degrees(ang)
        alineacion = "left"
        if 90 < rot < 270:
            rot += 180
            alineacion = "right"
        color_etiqueta = "black"
        if id_a_clase is not None and mapa_color_clase is not None:
            clase_muestra = id_a_clase.get(nombre)
            color_etiqueta = mapa_color_clase.get(clase_muestra, "black")
        ax.text(ang, d_max * 1.03, nombre, rotation=rot, rotation_mode="anchor",
                 ha=alineacion, va="center", fontsize=7, color=color_etiqueta,
                 weight="bold" if id_a_clase is not None else "normal")

    ax.set_ylim(0, d_max * 1.2)
    ax.set_yticklabels([])
    ax.spines["polar"].set_visible(False)
    ax.set_title("Circular dendrogram", y=1.08)
    fig.tight_layout()
    return fig


# =============================================================================
# PRETRATAMIENTOS ADICIONALES: RMN
# =============================================================================

def bucketing(X, eje, ancho_bucket):
    """
    Bucketing / binning: divide el eje espectral en cubetas de ancho fijo y
    suma (integra) la señal dentro de cada una. Reduce el número de variables
    y da tolerancia a pequeños corrimientos de pico entre muestras. Devuelve
    (X_bucketed, eje_bucketed) donde eje_bucketed es el centro de cada cubeta.

    'eje' debe estar ordenado de forma creciente o decreciente; se ordena
    internamente para el cálculo y se revierte al final si hacía falta.
    """
    eje = np.asarray(eje, dtype=float)
    X = np.asarray(X, dtype=float)
    orden = np.argsort(eje)
    eje_ord = eje[orden]
    X_ord = X[:, orden]

    bordes = np.arange(eje_ord[0], eje_ord[-1] + ancho_bucket, ancho_bucket)
    n_buckets = len(bordes) - 1
    if n_buckets < 1:
        raise ValueError("El ancho de bucket es mayor que el rango del eje espectral.")

    X_bucket = np.zeros((X.shape[0], n_buckets))
    centros = np.zeros(n_buckets)
    indices_bucket = np.clip(np.digitize(eje_ord, bordes) - 1, 0, n_buckets - 1)
    for b in range(n_buckets):
        mask = indices_bucket == b
        if mask.any():
            X_bucket[:, b] = X_ord[:, mask].sum(axis=1)
            centros[b] = eje_ord[mask].mean()
        else:
            centros[b] = (bordes[b] + bordes[b + 1]) / 2

    if eje[0] > eje[-1]:  # devolver en el mismo sentido que venía el eje original
        X_bucket = X_bucket[:, ::-1]
        centros = centros[::-1]
    return X_bucket, centros


def pqn(X, referencia=None):
    """
    Normalización por Cociente Probabilístico (PQN).
    Estima un factor de dilución por muestra (la mediana de los cocientes
    variable a variable contra una referencia) y divide cada espectro por su
    propio factor. Referencia por defecto: la mediana de todas las muestras.
    Más robusta que normalizar por área total frente a picos que cambian mucho.
    """
    X = np.asarray(X, dtype=float)
    if referencia is None:
        referencia = np.median(X, axis=0)
    referencia_segura = np.where(referencia == 0, 1e-12, referencia)
    cocientes = X / referencia_segura
    factores = np.median(cocientes, axis=1, keepdims=True)
    factores = np.where(factores == 0, 1e-12, factores)
    return X / factores


# =============================================================================
# PRETRATAMIENTOS ADICIONALES: CROMATOGRAMAS (y línea de base en general,
# también aplicable a Raman)
# =============================================================================

def linea_base_als(X, lam=1e5, p=0.01, n_iter=10):
    """
    Corrección de línea de base por Mínimos Cuadrados Asimétricos (ALS,
    Eilers & Boelens). Ajusta una línea de base suave, dando menor peso a los
    puntos que están claramente por encima de ella (los picos), y la resta.

    lam: controla qué tan suave es la línea de base (más alto = más suave).
    p: asimetría (más bajo = la línea de base "ignora" más los picos hacia
       arriba). Valores típicos: lam entre 1e4 y 1e8, p entre 0.001 y 0.1.
    """
    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    X = np.asarray(X, dtype=float)
    n_var = X.shape[1]
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(n_var, n_var - 2), dtype=float)
    D = lam * D.dot(D.transpose())

    X_corregido = np.zeros_like(X)
    for i in range(X.shape[0]):
        y = X[i, :]
        w = np.ones(n_var)
        for _ in range(n_iter):
            W = sparse.diags(w, 0, dtype=float)
            Z = (W + D).tocsc()
            z = spsolve(Z, w * y)
            w = p * (y > z) + (1 - p) * (y < z)
        X_corregido[i, :] = y - z
    return X_corregido


# =============================================================================
# PRETRATAMIENTOS ADICIONALES: RAMAN
# =============================================================================

def eliminar_rayos_cosmicos(X, ventana=5, umbral=7):
    """
    Elimina picos angostos y anómalamente intensos (rayos cósmicos), un
    artefacto típico de detectores CCD en Raman, no relacionado con la
    muestra. Para cada espectro, compara cada punto contra un filtro de
    mediana local; si se aparta demasiado (más de 'umbral' veces la
    desviación absoluta mediana del espectro), se reemplaza por el valor del
    filtro de mediana. 'ventana' debe ser impar.
    """
    from scipy.signal import medfilt

    X = np.asarray(X, dtype=float)
    X_limpio = X.copy()
    for i in range(X.shape[0]):
        mediana_movil = medfilt(X[i], kernel_size=ventana)
        diferencia = X[i] - mediana_movil
        mad = np.median(np.abs(diferencia - np.median(diferencia)))
        mad = mad if mad > 0 else 1e-12
        z_modificado = 0.6745 * diferencia / mad
        atipicos = np.abs(z_modificado) > umbral
        X_limpio[i, atipicos] = mediana_movil[atipicos]
    return X_limpio


# =============================================================================
# EXPORTACIÓN
# =============================================================================

def armar_dataframe_exportable(ids, eje, X):
    """
    Arma un DataFrame con el mismo formato que el archivo de entrada
    (ID de muestra como índice, números/longitudes de onda como columnas),
    listo para exportar a CSV/Excel un dataset ya transformado.
    """
    import pandas as pd
    df = pd.DataFrame(np.asarray(X), index=ids, columns=np.round(np.asarray(eje, dtype=float), 6))
    df.index.name = "id"
    return df


def interpolar_a_eje(X_nuevo, eje_nuevo, eje_objetivo):
    """
    Interpola cada espectro de X_nuevo (medido en eje_nuevo) al eje_objetivo
    (el eje con el que se entrenó un modelo). Necesario para predecir sobre
    muestras nuevas cuando el instrumento no midió exactamente los mismos
    números de onda que en el set de calibración.
    """
    eje_nuevo = np.asarray(eje_nuevo, dtype=float)
    eje_objetivo = np.asarray(eje_objetivo, dtype=float)
    X_nuevo = np.asarray(X_nuevo, dtype=float)

    orden = np.argsort(eje_nuevo)
    eje_ord, X_ord = eje_nuevo[orden], X_nuevo[:, orden]

    orden_obj = np.argsort(eje_objetivo)
    eje_obj_ord = eje_objetivo[orden_obj]

    fuera_de_rango = (eje_obj_ord.min() < eje_ord.min()) or (eje_obj_ord.max() > eje_ord.max())

    X_interp_ord = np.zeros((X_nuevo.shape[0], len(eje_obj_ord)))
    for i in range(X_nuevo.shape[0]):
        X_interp_ord[i] = np.interp(eje_obj_ord, eje_ord, X_ord[i])

    # Devolver en el mismo orden que venía eje_objetivo originalmente
    X_interp = np.zeros_like(X_interp_ord)
    X_interp[:, orden_obj] = X_interp_ord
    return X_interp, fuera_de_rango


# =============================================================================
# SIMCA (Soft Independent Modeling of Class Analogies)
# =============================================================================

def _calcular_q_cruzado(X_clase, n_comp, k_folds=None, random_state=0):
    """
    Computes an HONEST (cross-validated) Q residual for each calibration
    sample: for each fold, PCA is refit on the OTHER samples only, and the
    held-out samples are projected and their residual computed with that
    "unseen" model. In-sample (naive) residuals are always optimistically
    small, since PCA is fit to minimize exactly that residual on the
    calibration set — using them directly to set the class boundary makes
    it too tight for genuinely new samples. This is the standard fix.
    """
    from sklearn.model_selection import KFold
    n = X_clase.shape[0]
    k_folds = k_folds or min(10, n)
    if k_folds < 2:
        return None
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    Q_cv = np.zeros(n)
    for idx_fit, idx_val in kf.split(X_clase):
        if len(idx_fit) <= n_comp:
            return None
        pca_cv = PCA(n_components=n_comp).fit(X_clase[idx_fit])
        scores_val = pca_cv.transform(X_clase[idx_val])[:, :n_comp]
        Q_val = calcular_Q(X_clase[idx_val], scores_val, pca_cv.components_[:n_comp], n_comp, media=pca_cv.mean_)
        Q_cv[idx_val] = Q_val
    return Q_cv


def entrenar_modelo_simca(X_clase, n_comp=None, varianza_objetivo=0.95, alpha=0.05):
    """
    Fits a class-specific PCA model for SIMCA: a separate PCA model per
    class, used to test whether a new sample "belongs" to that class (based
    on its T² and Q distance to the class's own model), rather than
    discriminating directly between classes the way LDA/PLS-DA/etc. do. This
    is what lets SIMCA say "doesn't fit any known class" for a genuinely new
    kind of sample, instead of always forcing a pick among the classes it
    was trained on.

    If n_comp is None, picks the smallest number of components that reaches
    'varianza_objetivo' cumulative explained variance for this class alone.
    Returns a dict with everything needed to evaluate new samples later
    (see evaluar_muestras_simca).
    """
    n_muestras, n_variables = X_clase.shape
    max_comp = min(n_muestras - 1, n_variables)
    if max_comp < 1:
        raise ValueError("Not enough samples in this class to fit a SIMCA model (at least 2 are needed).")

    pca_completo = PCA(n_components=max_comp).fit(X_clase)
    if n_comp is None:
        var_acum = np.cumsum(pca_completo.explained_variance_ratio_)
        n_comp_calc = int(np.searchsorted(var_acum, varianza_objetivo) + 1)
    else:
        n_comp_calc = n_comp
    # Safety cap: using "too many" components relative to the number of
    # calibration samples leaves almost no residual degrees of freedom, which
    # makes Q look artificially tiny on the calibration set but then blow up
    # on genuinely new samples (the class model ends up fitting calibration
    # noise instead of real class structure). A common rule of thumb is to
    # keep at least 2/3 of the samples "in reserve" for the residual.
    limite_por_muestras = max(1, n_muestras // 3)
    n_comp_final = int(np.clip(n_comp_calc, 1, min(max_comp, limite_por_muestras)))

    scores = pca_completo.transform(X_clase)[:, :n_comp_final]
    cargas = pca_completo.components_[:n_comp_final, :]
    autovalores = pca_completo.explained_variance_[:n_comp_final]
    media = pca_completo.mean_

    T2_calibracion = calcular_T2(scores, autovalores, n_comp_final)
    T2_lim = limite_T2(n_muestras, n_comp_final, alpha)
    Q_calibracion = calcular_Q(X_clase, scores, cargas, n_comp_final, media=media)

    # Use cross-validated (honest) Q residuals to set the class boundary,
    # instead of the optimistically small in-sample residuals — this is what
    # actually lets the boundary generalize to genuinely new samples.
    Q_cv = _calcular_q_cruzado(X_clase, n_comp_final)
    Q_para_limite = Q_cv if Q_cv is not None else Q_calibracion
    Q_lim = limite_Q(autovalores, n_comp_final, alpha)
    if not limite_Q_confiable(Q_para_limite, Q_lim):
        Q_lim = np.percentile(Q_para_limite, 100 * (1 - alpha))
    else:
        # Even when the theoretical limit looks numerically plausible, still
        # widen it if it's tighter than what cross-validation suggests.
        Q_lim = max(Q_lim, np.percentile(Q_para_limite, 100 * (1 - alpha)))

    return {
        "media": media, "cargas": cargas, "autovalores": autovalores, "n_comp": n_comp_final,
        "T2_lim": T2_lim, "Q_lim": Q_lim,
        "T2_calibracion": T2_calibracion, "Q_calibracion": Q_calibracion,
        "Q_calibracion_cruzada": Q_cv,
        "varianza_explicada_pct": float(np.cumsum(pca_completo.explained_variance_ratio_)[n_comp_final - 1] * 100),
        "n_muestras_calibracion": n_muestras,
        "alpha": alpha,
        "limitado_por_muestras": n_comp_final < n_comp_calc,
    }


def evaluar_muestras_simca(X_nuevo, modelo_clase):
    """
    Projects new samples onto a class-specific SIMCA model (see
    entrenar_modelo_simca) and computes their T² and Q distance to that
    class. Returns (T2, Q, dentro_de_clase) — dentro_de_clase is True when
    the sample is within BOTH the T² and Q limits of this class model (the
    classic SIMCA acceptance rule); a sample can be "dentro" of zero, one,
    or several class models at once.
    """
    X_nuevo = np.asarray(X_nuevo, dtype=float)
    X_centrado = X_nuevo - modelo_clase["media"]
    scores = X_centrado @ modelo_clase["cargas"].T
    T2 = calcular_T2(scores, modelo_clase["autovalores"], modelo_clase["n_comp"])
    Q = calcular_Q(X_nuevo, scores, modelo_clase["cargas"], modelo_clase["n_comp"], media=modelo_clase["media"])
    dentro = (T2 <= modelo_clase["T2_lim"]) & (Q <= modelo_clase["Q_lim"])
    return T2, Q, dentro

# =============================================================================
# LEVERAGE / WILLIAMS PLOT (applicability domain diagnostic)
# =============================================================================

def calcular_leverage(X, n_comp=None, varianza_objetivo=0.95):
    """
    Leverage (hat value) of each sample, computed from a PCA fit on X — a
    generic "how far is this sample from the center of the data" measure
    that doesn't depend on which regression/classification algorithm is
    used downstream, so it can be paired with the residual of ANY model.
    """
    n = X.shape[0]
    max_comp = min(n - 1, X.shape[1])
    if max_comp < 1:
        return np.zeros(n)
    if n_comp is None:
        pca_completo = PCA(n_components=max_comp).fit(X)
        var_acum = np.cumsum(pca_completo.explained_variance_ratio_)
        n_comp = int(np.searchsorted(var_acum, varianza_objetivo) + 1)
    n_comp = int(np.clip(n_comp, 1, max_comp))
    pca = PCA(n_components=n_comp).fit(X)
    scores = pca.transform(X)
    # Normalize each component's scores by ITS OWN sum of squares (not the
    # grand total across all components) — this is what makes leverage
    # values sum to exactly n_comp + 1 across all samples, the standard
    # identity for a hat matrix.
    suma_cuadrados_por_componente = np.sum(scores ** 2, axis=0)
    suma_cuadrados_por_componente[suma_cuadrados_por_componente == 0] = 1.0
    leverage = 1.0 / n + np.sum(scores ** 2 / suma_cuadrados_por_componente[None, :], axis=1)
    return leverage, n_comp


def calcular_residuo_estandarizado(y_true, y_pred):
    """Standardized residual (y_true - y_pred) / std(residuals) — used on
    the Y-axis of a Williams plot alongside leverage on the X-axis."""
    residuos = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    desvio = np.std(residuos, ddof=1) if len(residuos) > 1 else 1.0
    return residuos / desvio if desvio > 0 else residuos


def limite_leverage(n_muestras, n_comp, factor=3):
    """
    Standard "warning leverage" threshold for a Williams plot:
    h* = factor * (n_comp + 1) / n_muestras (factor=3 is the usual QSAR
    convention; some references use 2 for a stricter cutoff).
    """
    return factor * (n_comp + 1) / n_muestras


# =============================================================================
# EMSC (Extended Multiplicative Scatter Correction)
# =============================================================================

def emsc(X, eje, x_ref=None, orden_polinomio=2):
    """
    Extended MSC: like MSC (additive offset + multiplicative scaling against
    a reference spectrum), but also models and removes wavelength-dependent
    effects (baseline drift, sloped scattering) using a polynomial of the
    given order in the (normalized) wavelength axis. More robust than plain
    MSC when scatter effects aren't perfectly wavelength-independent.
    """
    X = np.asarray(X, dtype=float)
    eje = np.asarray(eje, dtype=float)
    if x_ref is None:
        x_ref = X.mean(axis=0)

    desvio_eje = eje.std()
    eje_norm = (eje - eje.mean()) / (desvio_eje if desvio_eje > 0 else 1.0)

    columnas_diseno = [np.ones_like(eje_norm), x_ref]
    for grado in range(1, orden_polinomio + 1):
        columnas_diseno.append(eje_norm ** grado)
    M = np.column_stack(columnas_diseno)

    X_corregido = np.zeros_like(X)
    for i in range(X.shape[0]):
        coefs, _, _, _ = np.linalg.lstsq(M, X[i], rcond=None)
        b_i = coefs[1] if abs(coefs[1]) > 1e-8 else 1.0
        interferentes = M @ coefs - coefs[1] * x_ref
        X_corregido[i] = (X[i] - interferentes) / b_i
    return X_corregido


# =============================================================================
# MCR-ALS (Multivariate Curve Resolution — Alternating Least Squares)
# =============================================================================

def mcr_als(D, n_componentes, max_iter=100, tol=1e-6, no_negatividad=True, random_state=0):
    """
    Resolves a mixture data matrix D (samples x wavelengths) into pure
    component spectra (S) and concentration profiles (C), such that
    D ≈ C @ S, using alternating least squares. With non-negativity
    constraints (the default and usual choice for spectra/concentrations,
    which physically can't be negative).

    Returns a dict with C (n_samples x n_componentes), S (n_componentes x
    n_variables), lof_pct (lack of fit, %, lower is better) and n_iter.
    """
    from scipy.optimize import nnls

    D = np.asarray(D, dtype=float)
    n_muestras, n_variables = D.shape
    rng = np.random.RandomState(random_state)

    U, s_vals, Vt = np.linalg.svd(D, full_matrices=False)
    C = np.abs(U[:, :n_componentes] * s_vals[:n_componentes])
    if C.max() > 0:
        C = C / C.max()
    S_mat = np.abs(Vt[:n_componentes, :])

    lof_anterior = np.inf
    n_iter_usadas = 0
    for iteracion in range(max_iter):
        if no_negatividad:
            S_mat = np.array([nnls(C, D[:, j])[0] for j in range(n_variables)]).T
        else:
            S_mat, _, _, _ = np.linalg.lstsq(C, D, rcond=None)

        if no_negatividad:
            C = np.array([nnls(S_mat.T, D[i, :])[0] for i in range(n_muestras)])
        else:
            C_t, _, _, _ = np.linalg.lstsq(S_mat.T, D.T, rcond=None)
            C = C_t.T

        D_reconstruido = C @ S_mat
        residual = D - D_reconstruido
        lof = float(np.sqrt(np.sum(residual ** 2) / np.sum(D ** 2)) * 100) if np.sum(D ** 2) > 0 else 0.0
        n_iter_usadas = iteracion + 1
        if abs(lof_anterior - lof) < tol:
            break
        lof_anterior = lof

    return {"C": C, "S": S_mat, "lof_pct": lof, "n_iter": n_iter_usadas}
