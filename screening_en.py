"""
screening_en.py
Two independent pieces used by the app:

1. Spectral CROP helpers (mascara_recorte): which variables of an axis are kept
   when the user keeps / removes manually chosen regions.

2. MODEL SCREENING: quickly tries many combinations of
        spectrum (full / cropped) x preprocessing x variable selection x algorithm
   with deliberately cheap settings, and ranks them on the independent TEST set
   (AUC for classification, RMSE for regression, efficiency for SIMCA).

Methodological note: in the screening, variable selection (Boruta / genetic
algorithm) is fitted on the TRAINING samples only, and the same train/test split
is used for every combination, so the test ranking is comparable and not
inflated by selection done with test samples.
"""

import time
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression

import models_utils_en as mu
import chemo_utils as cu


# =============================================================================
# 1) CROP
# =============================================================================

def mascara_recorte(eje, crop):
    """Boolean mask over 'eje' of the variables to KEEP, or None if there is no
    effective crop. crop = {"modo": "keep"|"remove", "regiones": [(lo, hi), ...]}."""
    if not crop or not crop.get("regiones"):
        return None
    eje = np.asarray(eje, dtype=float)
    dentro = np.zeros(len(eje), dtype=bool)
    for lo, hi in crop["regiones"]:
        a, b = (lo, hi) if lo <= hi else (hi, lo)
        dentro |= (eje >= a) & (eje <= b)
    mascara = dentro if crop.get("modo", "keep") == "keep" else ~dentro
    if mascara.all() or mascara.sum() < 2:
        return None
    return mascara


def describir_recorte(crop):
    if not crop or not crop.get("regiones"):
        return None
    tramos = ", ".join(f"{min(a, b):g}–{max(a, b):g}" for a, b in crop["regiones"])
    return ("keep only " if crop.get("modo") == "keep" else "remove ") + tramos


# =============================================================================
# 2) PREPROCESSING RECIPES (same Step A / Step B vocabulary as the Preprocessing tab)
# =============================================================================

# (display name, Step A label, Step B label) — the first ones are pre-ticked
RECETAS_POR_TIPO = {
    "NIR / MIR": [
        ("None (no preprocessing)", "None", "None"),
        ("SNV", "SNV", "None"),
        ("MSC", "MSC", "None"),
        ("1st derivative", "None", "1st derivative"),
        ("2nd derivative", "None", "2nd derivative"),
        ("SNV + 1st derivative", "SNV", "1st derivative"),
        ("MSC + 1st derivative", "MSC", "1st derivative"),
        ("Smoothing", "None", "Smoothing (order 0)"),
        ("SNV + 2nd derivative", "SNV", "2nd derivative"),
        ("MSC + 2nd derivative", "MSC", "2nd derivative"),
        ("Standardization", "Standardization", "None"),
        ("Normalization (area)", "Normalization (area)", "None"),
    ],
    "Raman": [
        ("None (no preprocessing)", "None", "None"),
        ("Baseline correction (ALS)", "Baseline correction (ALS)", "None"),
        ("SNV", "SNV", "None"),
        ("Baseline (ALS) + 1st derivative", "Baseline correction (ALS)", "1st derivative"),
        ("SNV + 1st derivative", "SNV", "1st derivative"),
        ("Remove cosmic rays", "Remove cosmic rays", "None"),
        ("Baseline (ALS) + smoothing", "Baseline correction (ALS)", "Smoothing (order 0)"),
        ("Standardization", "Standardization", "None"),
    ],
    "Chromatogram": [
        ("None (no preprocessing)", "None", "None"),
        ("Baseline correction (ALS)", "Baseline correction (ALS)", "None"),
        ("Normalization (area)", "Normalization (area)", "None"),
        ("Standardization", "Standardization", "None"),
    ],
    "NMR": [
        ("None (no preprocessing)", "None", "None"),
        ("PQN normalization", "Normalization (PQN)", "None"),
        ("SNV", "SNV", "None"),
        ("Standardization", "Standardization", "None"),
    ],
}
PREMARCADAS = 7
RECETA_ACTUAL = "Current (as applied in the Preprocessing tab)"


def construir_secuencia(paso_a, paso_b, eje=None):
    """Steps (A then B) with sensible default parameters (SG window 11, order 2)."""
    mapa_a = {
        "Normalization (area)": ("normalizar", {"modo": "area"}),
        "Normalization (vector)": ("normalizar", {"modo": "vector"}),
        "Normalization (PQN)": ("pqn", {}),
        "Standardization": ("estandarizar", {}),
        "SNV": ("snv", {}),
        "MSC": ("msc", {}),
        "EMSC": ("emsc", {"eje": eje, "orden_polinomio": 2}),
        "Baseline correction (ALS)": ("linea_base_als", {"lam": 1e5, "p": 0.01}),
        "Remove cosmic rays": ("eliminar_rayos_cosmicos", {"ventana": 5, "umbral": 7}),
    }
    mapa_b = {
        "Smoothing (order 0)": ("suavizado_sg", {"ventana": 11, "orden_polinomio": 2}),
        "1st derivative": ("derivada_sg", {"orden": 1, "ventana": 11, "orden_polinomio": 2}),
        "2nd derivative": ("derivada_sg", {"orden": 2, "ventana": 11, "orden_polinomio": 2}),
    }
    return [p for p in (mapa_a.get(paso_a), mapa_b.get(paso_b)) if p is not None]


def contar_combinaciones(tarea, n_recetas, n_espectros, n_seleccion, n_algoritmos, n_varianzas=1):
    if tarea == "SIMCA":
        return n_recetas * n_espectros * n_varianzas
    return n_recetas * n_espectros * n_seleccion * n_algoritmos


# =============================================================================
# 3) THE SCREENING ENGINE
# =============================================================================

def _num(v):
    try:
        v = float(v)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _seleccionar(sel, Xtr, ytr, es_clf, cfg):
    """Variable selection fitted on TRAINING samples only. Returns bool mask or None."""
    if sel == "None":
        return None
    try:
        if sel == "Boruta":
            mascara = mu.seleccionar_variables_boruta(Xtr, ytr, es_clf, max_iter=int(cfg["boruta_iter"]))
        else:
            evaluador = LinearDiscriminantAnalysis() if es_clf else LinearRegression()
            ga = mu.SeleccionGenetica(
                evaluador, es_clasificacion=es_clf, tam_poblacion=int(cfg["ga_pob"]),
                n_generaciones=int(cfg["ga_gen"]), cv=3, random_state=0)
            ga.fit(Xtr, ytr)
            mascara = ga.mejor_mascara_
        mascara = np.asarray(mascara, dtype=bool)
        return mascara if mascara.sum() >= 2 else None
    except Exception:
        return None


def _fila_clasificacion(base, res, n_vars, hp, seg):
    cv, te = res["cv"], res.get("test", {})
    g = lambda d, k: _num(d.get(k)) if d else np.nan
    fila = dict(base)
    fila.update({
        "n variables": n_vars,
        "AUC (test)": g(te, "auc"), "Bal. accuracy (test)": g(te, "balanced_accuracy"),
        "Accuracy (test)": g(te, "accuracy"), "Sensitivity (test)": g(te, "sensibilidad_macro"),
        "Specificity (test)": g(te, "especificidad_macro"), "F1 macro (test)": g(te, "f1_macro"),
        "Kappa (test)": g(te, "kappa"), "MCC (test)": g(te, "mcc"),
        "AUC (CV)": g(cv, "auc"), "Bal. accuracy (CV)": g(cv, "balanced_accuracy"),
        "Accuracy (CV)": g(cv, "accuracy"), "Sensitivity (CV)": g(cv, "sensibilidad_macro"),
        "Specificity (CV)": g(cv, "especificidad_macro"), "F1 macro (CV)": g(cv, "f1_macro"),
        "Kappa (CV)": g(cv, "kappa"), "MCC (CV)": g(cv, "mcc"),
        "Hyperparameters": hp, "Time (s)": round(seg, 1),
    })
    return fila


def _fila_regresion(base, res, n_vars, hp, seg):
    cv, te = res["cv"], res.get("test", {})
    g = lambda d, k: _num(d.get(k)) if d else np.nan
    fila = dict(base)
    fila.update({
        "n variables": n_vars,
        "RMSE (test)": g(te, "rmse"), "R² (test)": g(te, "r2"), "RPD (test)": g(te, "rpd"),
        "RMSE (CV)": g(cv, "rmse_cv"), "R² (CV)": g(cv, "r2_cv"), "RPD (CV)": g(cv, "rpd_cv"),
        "RPIQ (CV)": g(cv, "rpiq_cv"),
        "Hyperparameters": hp, "Time (s)": round(seg, 1),
    })
    return fila


def ejecutar(tarea, datos, cfg, aplicar_paso, al_terminar_fila=None, progreso=None, ya_hechas=None):
    """
    tarea:  "Classification" | "Regression" | "SIMCA"
    datos:  dict with X_raw, eje_raw, X_actual, eje_actual (preprocessed as currently
            applied), ids, y (classes as text, or numeric reference values), crop (dict|None)
    cfg:    dict(recetas=[(nombre, a, b) | RECETA_ACTUAL], espectros=["Full","Cropped"],
                 seleccion=[...], algoritmos=[...], varianzas=[...], cv, prop_test,
                 optimizar(bool), boruta_iter, ga_pob, ga_gen)
    al_terminar_fila(fila): called after every combination (partial results survive an interruption)
    progreso(hechas, total, texto)
    ya_hechas: set of (receta, espectro, seleccion, algoritmo) to skip (to resume)
    """
    ya_hechas = ya_hechas or set()
    es_clf = tarea == "Classification"
    y = np.asarray(datos["y"])
    ids = np.asarray(datos["ids"])
    seleccion = cfg.get("seleccion", ["None"]) if tarea != "SIMCA" else ["-"]
    algoritmos = cfg.get("algoritmos", []) if tarea != "SIMCA" else ["SIMCA"]
    varianzas = cfg.get("varianzas", [0.95]) if tarea == "SIMCA" else [None]
    total = contar_combinaciones(tarea, len(cfg["recetas"]), len(cfg["espectros"]),
                                 len(seleccion), len(algoritmos), len(varianzas))
    hechas = 0
    filas = []
    catalogo = mu.crear_clasificadores() if es_clf else (mu.crear_regresores() if tarea == "Regression" else {})
    grillas = mu.GRILLAS_CLASIFICACION if es_clf else getattr(mu, "GRILLAS_REGRESION", {})
    prop = float(cfg["prop_test"])

    def avanzar(texto):
        if progreso:
            progreso(hechas, total, texto)

    for receta in cfg["recetas"]:
        nombre_rec = RECETA_ACTUAL if receta == RECETA_ACTUAL else receta[0]
        try:
            if receta == RECETA_ACTUAL:
                X_r, eje_r = np.asarray(datos["X_actual"], float), np.asarray(datos["eje_actual"], float)
            else:
                X_r, eje_r = np.asarray(datos["X_raw"], float), np.asarray(datos["eje_raw"], float)
                for paso in construir_secuencia(receta[1], receta[2], eje=eje_r):
                    X_r = aplicar_paso(X_r, paso)
        except Exception as e:
            for esp in cfg["espectros"]:
                for sel in seleccion:
                    for alg in algoritmos:
                        for _ in varianzas:
                            hechas += 1
                            fila = {"Spectrum": esp, "Preprocessing": nombre_rec, "Variable selection": sel,
                                    "Algorithm": alg, "Error": f"preprocessing failed: {e}"}
                            filas.append(fila)
                            if al_terminar_fila:
                                al_terminar_fila(fila)
            continue

        for esp in cfg["espectros"]:
            if esp == "Cropped":
                m = mascara_recorte(eje_r, datos.get("crop"))
                if m is None:
                    hechas += len(seleccion) * len(algoritmos) * len(varianzas)
                    continue
                X_e, eje_e = X_r[:, m], eje_r[m]
            else:
                X_e, eje_e = X_r, eje_r

            try:
                idx_tr, idx_te = mu.dividir_train_test(X_e, y, ids, prop, es_clf, 0, "random")
            except Exception as e:
                for sel in seleccion:
                    for alg in algoritmos:
                        for _ in varianzas:
                            hechas += 1
                            fila = {"Spectrum": esp, "Preprocessing": nombre_rec, "Variable selection": sel,
                                    "Algorithm": alg, "Error": f"split failed: {e}"}
                            filas.append(fila)
                            if al_terminar_fila:
                                al_terminar_fila(fila)
                continue

            # ---------------------------------------------------------------- SIMCA
            if tarea == "SIMCA":
                for var in varianzas:
                    clave = (nombre_rec, esp, "-", f"SIMCA {var:.0%}")
                    hechas += 1
                    if clave in ya_hechas:
                        continue
                    avanzar(f"{nombre_rec} · {esp} · SIMCA ({var:.0%} variance)")
                    t0 = time.perf_counter()
                    base = {"Spectrum": esp, "Preprocessing": nombre_rec, "Variable selection": "-",
                            "Algorithm": f"SIMCA {var:.0%}"}
                    try:
                        modelos = {c: cu.entrenar_modelo_simca(X_e[idx_tr][y[idx_tr] == c],
                                                               varianza_objetivo=var, alpha=0.05)
                                   for c in np.unique(y)}
                        Xte, yte = X_e[idx_te], y[idx_te]
                        dentro = {c: cu.evaluar_muestras_simca(Xte, m_)[2] for c, m_ in modelos.items()}
                        sens, espf = [], []
                        for c in modelos:
                            miembro = yte == c
                            if miembro.any():
                                sens.append(dentro[c][miembro].mean())
                            if (~miembro).any():
                                espf.append((~dentro[c][~miembro]).mean())
                        s_m, e_m = float(np.mean(sens)), float(np.mean(espf))
                        exacto = np.mean([
                            {c for c in modelos if dentro[c][i]} == {yte[i]} for i in range(len(yte))])
                        fila = dict(base)
                        fila.update({
                            "n variables": X_e.shape[1],
                            "Efficiency (test)": float(np.sqrt(s_m * e_m)),
                            "Sensitivity (test)": s_m, "Specificity (test)": e_m,
                            "Exact assignment (test)": float(exacto),
                            "Avg. components": float(np.mean([m_["n_comp"] for m_ in modelos.values()])),
                            "Time (s)": round(time.perf_counter() - t0, 1)})
                    except Exception as e:
                        fila = dict(base)
                        fila["Error"] = str(e)[:150]
                    filas.append(fila)
                    if al_terminar_fila:
                        al_terminar_fila(fila)
                continue

            # ------------------------------------------------- classification / regression
            for sel in seleccion:
                pendientes = [a for a in algoritmos if (nombre_rec, esp, sel, a) not in ya_hechas]
                if not pendientes:
                    hechas += len(algoritmos)
                    continue
                avanzar(f"{nombre_rec} · {esp} · variable selection: {sel}")
                mascara = _seleccionar(sel, X_e[idx_tr], y[idx_tr], es_clf, cfg)
                X_s = X_e[:, mascara] if mascara is not None else X_e
                for alg in algoritmos:
                    clave = (nombre_rec, esp, sel, alg)
                    if clave in ya_hechas:
                        hechas += 1
                        continue
                    hechas += 1
                    avanzar(f"{nombre_rec} · {esp} · {sel} · {alg}")
                    t0 = time.perf_counter()
                    base = {"Spectrum": esp, "Preprocessing": nombre_rec,
                            "Variable selection": sel, "Algorithm": alg}
                    try:
                        modelo, hp = catalogo[alg], ""
                        if cfg.get("optimizar") and alg in grillas:
                            modelo, mejores, _, _ = mu.optimizar_hiperparametros(
                                modelo, grillas[alg], X_s[idx_tr], y[idx_tr],
                                es_clasificacion=es_clf, cv=int(cfg["cv"]), metodo="fast")
                            hp = ", ".join(f"{k}={v}" for k, v in mejores.items())
                        func = mu.entrenar_evaluar_clasificacion if es_clf else mu.entrenar_evaluar_regresion
                        res = func(modelo, X_s, y, ids=ids, cv=int(cfg["cv"]),
                                   proporcion_test=prop, metodo_split="random")
                        fila = (_fila_clasificacion if es_clf else _fila_regresion)(
                            base, res, X_s.shape[1], hp, time.perf_counter() - t0)
                    except Exception as e:
                        fila = dict(base)
                        fila["Error"] = str(e)[:150]
                    filas.append(fila)
                    if al_terminar_fila:
                        al_terminar_fila(fila)
    if progreso:
        progreso(total, total, "done")
    return filas


# =============================================================================
# 4) RANKING TABLE
# =============================================================================

COLUMNAS_INICIALES = ["Rank", "Spectrum", "Preprocessing", "Variable selection", "Algorithm", "n variables"]
CLAVE_RANKING = {
    "Classification": ("AUC (test)", False, "Bal. accuracy (test)"),
    "Regression": ("RMSE (test)", True, "R² (test)"),
    "SIMCA": ("Efficiency (test)", False, "Exact assignment (test)"),
}


def tabla_ranking(filas, tarea):
    """DataFrame sorted best-first on the test metric (AUC / RMSE / efficiency), with
    every other training (CV) and test metric alongside. Failed combinations go last."""
    if not filas:
        return pd.DataFrame()
    df = pd.DataFrame(filas)
    principal, ascendente, desempate = CLAVE_RANKING[tarea]
    if principal not in df.columns:
        df[principal] = np.nan
    if desempate not in df.columns:
        df[desempate] = np.nan
    fallback = {"Classification": "AUC (CV)", "Regression": "RMSE (CV)"}.get(tarea)
    clave = df[principal].copy()
    if fallback and fallback in df.columns:      # no usable test value -> use the CV one
        clave = clave.fillna(df[fallback])
    df["_k1"] = clave
    df["_k2"] = df[desempate]
    # main test metric, then the tie-breaker (always "higher is better"), then fewer
    # variables (a simpler model wins a tie)
    nv = df["n variables"] if "n variables" in df.columns else 0
    df["_k3"] = nv
    df = df.sort_values(["_k1", "_k2", "_k3"], ascending=[ascendente, False, True],
                        na_position="last", kind="mergesort").drop(columns=["_k1", "_k2", "_k3"])
    df.insert(0, "Rank", range(1, len(df) + 1))
    orden = [c for c in COLUMNAS_INICIALES if c in df.columns]
    resto = [c for c in df.columns if c not in orden and c not in ("Error",)]
    cols = orden + [principal] + [c for c in resto if c != principal]
    if "Error" in df.columns:
        cols.append("Error")
    return df[cols].reset_index(drop=True)
