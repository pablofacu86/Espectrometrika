"""
informe_final_en.py
Builds the combined "Final report" from the results currently stored in the
app's session state (PCA, outliers, dendrogram, the 'Other tools', classification,
SIMCA, regression, the last prediction and model cards).

It never recomputes anything: it only lays out what the user already computed,
and flags any result that no longer matches the current data / preprocessing /
excluded outliers (so an out-of-date result can never be mistaken for a current one).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import reporte_utils_en as ru
import chemo_utils as cu

# (group title, [(key, label), ...]) — order = order in the PDF
GRUPOS = [
    ("Data", [("dataset", "Dataset and preprocessing summary")]),
    ("Exploratory analysis", [
        ("pca", "PCA"),
        ("outliers", "Outlier detection (Hotelling's T² / Q)"),
        ("dendro", "Hierarchical clustering (dendrogram)"),
    ]),
    ("Other exploratory tools", [
        ("otros_loadings", "Loadings vs wavenumber"),
        ("otros_ranking", "Ranking of important variables"),
        ("otros_corr", "Loadings correlation plot"),
        ("otros_media", "Mean spectrum by class"),
        ("otros_cm", "Clustermap"),
        ("otros_tsne", "t-SNE"),
        ("otros_umap", "UMAP"),
        ("otros_mcr", "MCR-ALS"),
    ]),
    ("Models", [
        ("clf", "Classification"),
        ("simca", "SIMCA"),
        ("reg", "Regression"),
    ]),
    ("Prediction", [("pred", "Last prediction on new samples")]),
]
TODAS_LAS_CLAVES = [k for _, items in GRUPOS for k, _ in items]
ETIQUETAS = {k: l for _, items in GRUPOS for k, l in items}


# =============================================================================
# AVAILABILITY / STALENESS
# =============================================================================

def _hay(ss, clave):
    if clave == "dataset":
        return ss.get("X") is not None
    if clave == "pca":
        return ss.get("pca_completo") is not None
    if clave == "outliers":
        return ss.get("outliers_resultado") is not None
    if clave == "dendro":
        return ss.get("dendro_resultado") is not None
    if clave.startswith("otros_"):
        return ss.get(f"{clave}_resultado") is not None
    if clave == "clf":
        return ss.get("clf_resultados") is not None
    if clave == "simca":
        return bool(ss.get("simca_modelos"))
    if clave == "reg":
        return ss.get("reg_resultados") is not None
    if clave == "pred":
        return ss.get("pred_ultimo") is not None
    return False


def _desactualizado(ss, clave, firma_actual):
    """True if the stored result was computed from a different data/preprocessing/
    outlier state than the current one (it keeps showing the old result)."""
    try:
        if clave == "pca":
            f = ss.get("pca_firma")
            return f is not None and f != firma_actual
        if clave == "outliers":
            f = ss.get("outliers_firma")
            return f is not None and tuple(f)[:len(firma_actual)] != tuple(firma_actual)
        if clave in ("otros_loadings", "otros_ranking", "otros_corr"):
            f = ss.get(f"{clave}_firma")
            return f is not None and f[1] != ss.get("pca_firma")
        if clave in ("dendro", "otros_media", "otros_cm", "otros_tsne", "otros_umap", "otros_mcr"):
            f = ss.get(f"{clave}_firma")
            return f is not None and f[0] != firma_actual
        if clave in ("clf", "reg"):
            f = ss.get(f"{clave}_firma")
            return f is not None and f != firma_actual
    except Exception:
        return False
    return False


def disponibilidad(ss, firma_actual):
    """{key: {"disponible": bool, "desactualizado": bool}} for every section."""
    res = {}
    for k in TODAS_LAS_CLAVES:
        disp = _hay(ss, k)
        res[k] = {"disponible": disp, "desactualizado": disp and _desactualizado(ss, k, firma_actual)}
    return res


# =============================================================================
# SMALL HELPERS
# =============================================================================

def _como_texto(clases):
    return None if clases is None else np.array([str(c) for c in clases])


def _paleta(n):
    cmap = plt.get_cmap("tab10")
    return [cmap(i % 10) for i in range(n)]


def _fig_lineas(eje, curvas, etiquetas, titulo, ylabel="Signal"):
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for y, et, col in zip(curvas, etiquetas, _paleta(len(curvas))):
        ax.plot(eje, y, label=et, color=col, linewidth=1.1)
    ax.set_xlabel("Wavenumber")
    ax.set_ylabel(ylabel)
    ax.set_title(titulo, fontsize=10)
    if len(etiquetas) > 1:
        ax.legend(fontsize=8, frameon=False)
    if len(eje) and eje[0] > eje[-1]:
        ax.invert_xaxis()
    fig.tight_layout()
    return fig


def _fig_scatter2d(emb, clases, titulo, xl, yl):
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    if clases is None:
        ax.scatter(emb[:, 0], emb[:, 1], s=22, color="#185FA5", alpha=0.85)
    else:
        for c, col in zip(list(dict.fromkeys(clases)), _paleta(len(set(clases)))):
            m = clases == c
            ax.scatter(emb[m, 0], emb[m, 1], s=24, label=str(c), color=col, alpha=0.85)
        ax.legend(fontsize=8, frameon=False)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.set_title(titulo, fontsize=10)
    fig.tight_layout()
    return fig


def _stale(desact):
    if not desact:
        return []
    return [{"tipo": "parrafo", "texto":
             "WARNING - OUT OF DATE: this result was computed with a data / preprocessing / "
             "excluded-outliers state that is different from the one currently applied in the "
             "app, and was not updated. Treat it as the result for the older state."}]


# =============================================================================
# SECTION BUILDERS (each returns a list of report blocks, without the title)
# =============================================================================

def _b_dataset(ss):
    X = ss["X"]
    mascara = ss.get("mascara_excluidas")
    n_excl = int(mascara.sum()) if mascara is not None else 0
    ids = np.array(ss["ids"])
    pares = [
        ("File", ss.get("archivo_cargado_nombre", "n/a")),
        ("Samples loaded", int(X.shape[0])),
        ("Variables (original axis)", int(X.shape[1])),
        ("Samples excluded as outliers", n_excl),
        ("Preprocessing applied to the analysis", ss.get("pret_desc_aplicada", "none (raw spectra)")),
    ]
    clases = ss.get("clases")
    if clases is not None:
        ct = pd.Series(_como_texto(clases)).value_counts()
        pares.insert(3, ("Classes (samples per class)", ", ".join(f"{k}: {v}" for k, v in ct.items())))
    bloques = [{"tipo": "clave_valor", "pares": pares}]
    if n_excl:
        bloques.append({"tipo": "parrafo", "texto":
                        "Excluded samples: " + ", ".join(map(str, ids[mascara][:60])) +
                        (" ..." if n_excl > 60 else "")})
    try:
        activo = ~mascara if mascara is not None else np.ones(X.shape[0], dtype=bool)
        cl = _como_texto(clases[activo]) if clases is not None else None
        bloques.append({"tipo": "imagen", "fig": ru.fig_espectros(
            ss["numeros_onda_pret"], ss["X_pret"][activo], clases=cl,
            titulo="Spectra used by the analysis (after the applied preprocessing)")})
    except Exception:
        pass
    return bloques


def _b_pca(ss, desact):
    sc, var = ss["scores_completo"], ss["var_explicada"]
    ids, clases = ss["pca_ids"], ss["pca_clases"]
    n_comp = int(min(ss.get("n_comp", 3), len(var)))
    bloques = _stale(desact) + [{"tipo": "clave_valor", "pares": [
        ("Samples used", len(ids)),
        ("Variables", int(ss["pca_X_input"].shape[1])),
        ("Components retained", n_comp),
        ("Cumulative explained variance", f"{np.cumsum(var)[n_comp - 1]:.1f}%"),
    ]}]
    bloques.append({"tipo": "imagen", "fig": ru.fig_scree(var, n_comp_marcado=n_comp)})
    bloques.append({"tipo": "imagen", "fig": ru.fig_scores(sc, var, 1, 2, clases=clases, ids=ids)})
    if sc.shape[1] >= 3:
        bloques.append({"tipo": "imagen", "fig": ru.fig_scores(sc, var, 1, 3, clases=clases, ids=ids)})
    return bloques


def _b_outliers(ss, desact):
    T2, T2_lim, Q, Q_lim, confiable = ss["outliers_resultado"]
    ids, clases = np.array(ss["outliers_ids"]), ss["outliers_clases"]
    f = ss["outliers_firma"]
    alpha, n_comp = f[-1], f[-2]
    ambos = (T2 > T2_lim) & (Q > Q_lim)
    solo_t2 = (T2 > T2_lim) & ~(Q > Q_lim)
    solo_q = (Q > Q_lim) & ~(T2 > T2_lim)
    T2_lo, T2_hi = cu.rango_con_margen(T2, T2_lim)
    Q_lo, Q_hi = cu.rango_con_margen(Q, Q_lim)
    excl = ss.get("mascara_excluidas")
    bloques = _stale(desact) + [{"tipo": "clave_valor", "pares": [
        ("Samples analysed", len(ids)),
        ("Principal components used", n_comp),
        ("Significance level (alpha)", alpha),
        ("T² limit", f"{T2_lim:.2f}"),
        ("Q limit", f"{Q_lim:.4g}" + (" (empirical)" if not confiable else " (theoretical)")),
        ("Above BOTH limits", f"{int(ambos.sum())}: " + (", ".join(ids[ambos]) if ambos.any() else "none")),
        ("Above the T² limit only", f"{int(solo_t2.sum())}: " + (", ".join(ids[solo_t2]) if solo_t2.any() else "none")),
        ("Above the Q limit only", f"{int(solo_q.sum())}: " + (", ".join(ids[solo_q]) if solo_q.any() else "none")),
        ("Currently excluded from the analysis", int(excl.sum()) if excl is not None else 0),
    ]}]
    bloques.append({"tipo": "imagen", "fig": ru.fig_outliers(
        T2, Q, T2_lim, Q_lim, T2_lo, T2_hi, Q_lo, Q_hi, clases=_como_texto(clases))})
    return bloques


def _b_dendro(ss, desact):
    r = ss["dendro_resultado"]
    return _stale(desact) + [
        {"tipo": "clave_valor", "pares": [("Linkage method", r["metodo"]), ("Samples", len(r["ids"]))]},
        {"tipo": "imagen", "fig": ru.fig_dendrograma(r["Z"], list(r["ids"]), clases=_como_texto(r["clases"]))},
    ]


def _b_otros(ss, clave, desact):
    r = ss[f"{clave}_resultado"]
    b = _stale(desact)
    if clave == "otros_loadings":
        cargas, eje = r["cargas"], r["eje"]
        n = min(3, cargas.shape[0])
        b.append({"tipo": "imagen", "fig": _fig_lineas(
            eje, [cargas[i] for i in range(n)], [f"PC{i+1}" for i in range(n)],
            "Loadings of the first principal components", ylabel="Loading")})
    elif clave == "otros_ranking":
        cargas, eje = r["cargas"], r["eje"]
        orden = np.argsort(np.abs(cargas[0]))[::-1][:15]
        b.append({"tipo": "parrafo", "texto": "The 15 variables with the largest absolute loading on PC1:"})
        b.append({"tipo": "tabla", "encabezados": ["Rank", "Wavenumber", "Loading on PC1"],
                  "filas": [[k + 1, f"{eje[i]:.2f}", f"{cargas[0, i]:.4f}"] for k, i in enumerate(orden)]})
    elif clave == "otros_corr":
        cargas, auto, eje = r["cargas"], r["autovalores"], r["eje"]
        lx, ly = cargas[0] * np.sqrt(auto[0]), cargas[1] * np.sqrt(auto[1])
        esc = 1 / max(np.max(np.abs(lx)), np.max(np.abs(ly)))
        lx, ly = lx * esc, ly * esc
        top = np.argsort(lx ** 2 + ly ** 2)[-12:]
        fig, ax = plt.subplots(figsize=(5.6, 5.2))
        ax.scatter(lx, ly, s=8, color="#185FA5", alpha=0.35)
        ax.scatter(lx[top], ly[top], s=22, color="darkred")
        for i in top:
            ax.annotate(f"{eje[i]:.0f}", (lx[i], ly[i]), fontsize=7, xytext=(2, 3), textcoords="offset points")
        t = np.linspace(0, 2 * np.pi, 100)
        for rad in (0.5, 1.0):
            ax.plot(rad * np.cos(t), rad * np.sin(t), "--", color="gray", linewidth=0.8)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect("equal")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("Loadings correlation plot (PC1 vs PC2)", fontsize=10)
        fig.tight_layout()
        b.append({"tipo": "imagen", "fig": fig, "ancho_mm": 120})
    elif clave == "otros_media":
        eje = r["eje"]
        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        for (nombre, (m, s)), col in zip(r["por_clase"].items(), _paleta(len(r["por_clase"]))):
            ax.plot(eje, m, label=nombre, color=col, linewidth=1.2)
            ax.fill_between(eje, m - s, m + s, color=col, alpha=0.15, linewidth=0)
        ax.set_xlabel("Wavenumber")
        ax.set_ylabel("Signal")
        ax.set_title("Mean spectrum of each class (± 1 standard deviation)", fontsize=10)
        ax.legend(fontsize=8, frameon=False)
        if eje[0] > eje[-1]:
            ax.invert_xaxis()
        fig.tight_layout()
        b.append({"tipo": "imagen", "fig": fig})
    elif clave == "otros_cm":
        b.append({"tipo": "parrafo", "texto": f"Linkage method: {r['metodo']}."})
        b.append({"tipo": "imagen_png", "png": r["png"]})
    elif clave in ("otros_tsne", "otros_umap"):
        nom = "t-SNE" if clave == "otros_tsne" else "UMAP"
        b.append({"tipo": "imagen", "fig": _fig_scatter2d(
            r["emb"], _como_texto(r["clases"]), f"{nom} map", f"{nom} 1", f"{nom} 2"), "ancho_mm": 130})
    elif clave == "otros_mcr":
        res, ids, eje = r["res"], np.array(r["ids"]), r["eje"]
        b.append({"tipo": "clave_valor", "pares": [
            ("Components resolved", r["n"]), ("Lack of fit", f"{res['lof_pct']:.2f}%"),
            ("Iterations to converge", res["n_iter"])]})
        b.append({"tipo": "imagen", "fig": _fig_lineas(
            eje, [res["S"][k] for k in range(res["S"].shape[0])],
            [f"Component {k+1}" for k in range(res["S"].shape[0])],
            "Resolved pure-component spectra", ylabel="Signal (a.u.)")})
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        base = np.zeros(len(ids))
        for k, col in zip(range(res["C"].shape[1]), _paleta(res["C"].shape[1])):
            ax.bar(range(len(ids)), res["C"][:, k], bottom=base, label=f"Component {k+1}", color=col)
            base = base + res["C"][:, k]
        ax.set_xticks(range(len(ids)))
        ax.set_xticklabels(ids, rotation=90, fontsize=5 if len(ids) > 60 else 7)
        ax.set_ylabel("Relative concentration")
        ax.set_title("Concentration profile per sample", fontsize=10)
        ax.legend(fontsize=8, frameon=False)
        fig.tight_layout()
        b.append({"tipo": "imagen", "fig": fig})
    return b


def _hiperparametros(ss, prefijo, nombre):
    hp = ss.get(f"{prefijo}_hiperparametros", {}).get(nombre)
    if not hp:
        return []
    return [{"tipo": "subtitulo", "texto": "Optimized hyperparameters"},
            {"tipo": "clave_valor", "pares": [
                ("Optimization method", ss.get(f"{prefijo}_descripcion_opt", {}).get(nombre, "n/a"))]
                + list(hp.items())}]


def _b_clf(ss, desact, detalle):
    res_all = ss["clf_resultados"]
    cv = ss.get("clf_cv_folds")
    b = _stale(desact) + [{"tipo": "clave_valor", "pares": [
        ("Samples used", ss.get("clf_n_muestras", "n/a")),
        ("Cross-validation", f"{cv}-fold" if cv != "LOO" else "Leave-One-Out"),
        ("Independent test proportion", f"{int(ss.get('clf_prop_test', 0) * 100)}% ({ss.get('clf_metodo_split', 'n/a')})"),
        ("Variable selection", ss.get("clf_metodo_seleccion", "None")),
    ]}]
    mv = ss.get("clf_mascara_variables")
    if mv is not None:
        b.append({"tipo": "parrafo", "texto": f"Selected variables: {int(mv.sum())} of {len(mv)}."})
        try:
            b.append({"tipo": "imagen", "fig": ru.fig_variables_seleccionadas(
                ss["clf_eje_usado"], ss["clf_espectro_promedio"], mv)})
        except Exception:
            pass
    filas = []
    for nombre, res in res_all.items():
        if "error" in res:
            filas.append([nombre, "ERROR: " + str(res["error"])[:80]] + [""] * 0)
            continue
        f = [nombre] + [round(res["cv"][k], 3) for k in
                        ("accuracy", "balanced_accuracy", "sensibilidad_macro", "especificidad_macro",
                         "f1_macro", "kappa", "mcc")]
        f.append(round(res["cv"]["auc"], 3) if res["cv"].get("auc") is not None else "n/a")
        if "test" in res:
            f += [round(res["test"][k], 3) for k in ("accuracy", "sensibilidad_macro", "especificidad_macro", "kappa", "mcc")]
        else:
            f += ["-"] * 5
        filas.append(f)
    enc = ["Model", "Accuracy (CV)", "Bal. Accuracy (CV)", "Sensitivity (CV)", "Specificity (CV)",
           "F1 macro (CV)", "Kappa (CV)", "MCC (CV)", "AUC (CV)", "Accuracy (test)", "Sensitivity (test)",
           "Specificity (test)", "Kappa (test)", "MCC (test)"]
    filas = [f if len(f) == len(enc) else f + [""] * (len(enc) - len(f)) for f in filas]
    b += [{"tipo": "subtitulo", "texto": "Model comparison"}, {"tipo": "tabla", "encabezados": enc, "filas": filas}]
    for nombre in detalle:
        res = res_all.get(nombre)
        if not res or "error" in res:
            continue
        b.append({"tipo": "subtitulo", "texto": f"Detail: {nombre}"})
        b += _hiperparametros(ss, "clf", nombre)
        clases_orden = list(res["cv"]["clases"])
        b.append({"tipo": "imagen", "fig": ru.fig_matriz_confusion(res["cv"]["matriz_confusion"], clases_orden),
                  "ancho_mm": 120})
        if "test" in res:
            M = confusion_matrix(res["test"]["y_true"], res["test"]["y_pred"], labels=clases_orden)
            b.append({"tipo": "parrafo", "texto": "Confusion matrix on the independent test set:"})
            b.append({"tipo": "imagen", "fig": ru.fig_matriz_confusion(M, clases_orden), "ancho_mm": 120})
        if ss.get("clf_curva_modelo") == nombre and ss.get("clf_curva_aprendizaje") is not None:
            b.append({"tipo": "imagen", "fig": ru.fig_curva_aprendizaje(ss["clf_curva_aprendizaje"], nombre)})
    return b


def _b_simca(ss):
    modelos = ss["simca_modelos"]
    matriz, true_ = ss["simca_matriz_dentro"], np.array(ss["simca_clases_eval_true"])
    n_acc = np.array(ss["simca_n_clases_aceptado"])
    b = [{"tipo": "clave_valor", "pares": [
        ("Samples used", ss.get("simca_n_muestras", "n/a")),
        ("Significance level (alpha)", ss.get("simca_alpha", "n/a")),
        ("Target explained variance", ss.get("simca_varianza_objetivo", "n/a")),
        ("Test proportion", f"{int(ss.get('simca_prop_test', 0) * 100)}% ({ss.get('simca_metodo_split', 'n/a')})"),
    ]}]
    if ss.get("simca_validacion_es_calibracion"):
        b.append({"tipo": "parrafo", "texto":
                  "NOTE: no independent test set was used - the figures below were measured on the same "
                  "samples used to fit each class model, so they are optimistic."})
    filas = []
    for c, m in modelos.items():
        miembro = true_ == c
        dentro = matriz[c].to_numpy()
        sens = dentro[miembro].mean() if miembro.any() else np.nan
        espec = (~dentro[~miembro]).mean() if (~miembro).any() else np.nan
        filas.append([c, m["n_comp"], f"{m['varianza_explicada_pct']:.1f}%", m["n_muestras_calibracion"],
                      f"{sens:.3f}" if pd.notna(sens) else "n/a", f"{espec:.3f}" if pd.notna(espec) else "n/a"])
    b.append({"tipo": "tabla", "encabezados": ["Class", "Components", "Variance explained", "Calibration samples",
                                               "Sensitivity (members accepted)", "Specificity (non-members rejected)"],
              "filas": filas})
    b.append({"tipo": "clave_valor", "pares": [
        ("Accepted by exactly 1 class", int((n_acc == 1).sum())),
        ("Accepted by 2+ classes (ambiguous)", int((n_acc >= 2).sum())),
        ("Rejected by all classes (outlier)", int((n_acc == 0).sum()))]})
    return b


def _b_reg(ss, desact, detalle):
    res_all = ss["reg_resultados"]
    cv = ss.get("reg_cv_folds")
    b = _stale(desact) + [{"tipo": "clave_valor", "pares": [
        ("Samples used", ss.get("reg_n_muestras", "n/a")),
        ("Cross-validation", f"{cv}-fold" if cv != "LOO" else "Leave-One-Out"),
        ("Independent test proportion", f"{int(ss.get('reg_prop_test', 0) * 100)}% ({ss.get('reg_metodo_split', 'n/a')})"),
        ("Variable selection", ss.get("reg_metodo_seleccion", "None")),
    ]}]
    mv = ss.get("reg_mascara_variables")
    if mv is not None:
        b.append({"tipo": "parrafo", "texto": f"Selected variables: {int(mv.sum())} of {len(mv)}."})
    enc = ["Model", "RMSE (CV)", "R2 (CV)", "RPD (CV)", "RPIQ (CV)", "RMSE (test)", "R2 (test)", "RPD (test)"]
    filas = []
    for nombre, res in res_all.items():
        if "error" in res:
            filas.append([nombre, "ERROR: " + str(res["error"])[:80]] + [""] * 6)
            continue
        f = [nombre, round(res["cv"]["rmse_cv"], 4), round(res["cv"]["r2_cv"], 3),
             round(res["cv"]["rpd_cv"], 2), round(res["cv"]["rpiq_cv"], 2)]
        f += ([round(res["test"]["rmse"], 4), round(res["test"]["r2"], 3), round(res["test"]["rpd"], 2)]
              if "test" in res else ["-"] * 3)
        filas.append(f)
    b += [{"tipo": "subtitulo", "texto": "Model comparison"}, {"tipo": "tabla", "encabezados": enc, "filas": filas}]
    for nombre in detalle:
        res = res_all.get(nombre)
        if not res or "error" in res:
            continue
        b.append({"tipo": "subtitulo", "texto": f"Detail: {nombre}"})
        b.append({"tipo": "parrafo", "texto":
                  f"Approximate 95% prediction interval for new samples: prediction +/- "
                  f"{1.96 * res['cv']['rmse_cv']:.4g} (1.96 x the cross-validation RMSE of {res['cv']['rmse_cv']:.4g})."})
        b += _hiperparametros(ss, "reg", nombre)
        b.append({"tipo": "parrafo", "texto": "Predicted vs. actual (cross-validation):"})
        b.append({"tipo": "imagen", "fig": ru.fig_predicho_vs_real(res["cv"]["y_true"], res["cv"]["y_pred"]),
                  "ancho_mm": 130})
        if "test" in res:
            b.append({"tipo": "parrafo", "texto": "Predicted vs. actual (independent test set):"})
            b.append({"tipo": "imagen", "fig": ru.fig_predicho_vs_real(res["test"]["y_true"], res["test"]["y_pred"]),
                      "ancho_mm": 130})
        if ss.get("reg_curva_modelo") == nombre and ss.get("reg_curva_aprendizaje") is not None:
            b.append({"tipo": "imagen", "fig": ru.fig_curva_aprendizaje(ss["reg_curva_aprendizaje"], nombre)})
    return b


def _b_pred(ss):
    p = ss["pred_ultimo"]
    df = p["df"]
    b = [{"tipo": "clave_valor", "pares": [
        ("Model used", p["modelo"]), ("File with the new samples", p["archivo"]),
        ("Number of samples", len(df)), ("Predicted on", p.get("fecha", "n/a"))]}]
    ficha = p.get("ficha")
    if ficha:
        b.append({"tipo": "parrafo", "texto":
                  f"Model traceability ID: {ficha.get('id_trazabilidad', 'n/a')} (trained on "
                  f"{ficha.get('fecha_creacion', 'n/a')}; preprocessing: {ficha.get('pretratamiento_desc', 'n/a')})."})
    if "lower_95" in df.columns:
        b.append({"tipo": "parrafo", "texto":
                  "lower_95 / upper_95 is an approximate 95% prediction interval built from the model's "
                  "cross-validation error; treat it with extra caution near the edges of the calibration range."})
    if "confidence" in df.columns:
        b.append({"tipo": "parrafo", "texto":
                  "confidence is the model's probability for its own predicted class."})
    b.append({"tipo": "tabla", "encabezados": list(df.columns),
              "filas": df.astype(str).values.tolist()[:80]})
    if len(df) > 80:
        b.append({"tipo": "parrafo", "texto":
                  f"(showing the first 80 of {len(df)} predictions - the full table is in the Excel file "
                  "downloadable from the Prediction tab)"})
    return b


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def construir_informe(ss, opciones, firma_actual):
    """
    opciones: {"titulo": str, "notas": str, "incluir": set(keys),
               "clf_detalle": [...], "reg_detalle": [...], "fichas": [saved model names]}
    Returns (titulo, subtitulo, bloques, indice) where 'indice' lists the included sections.
    """
    incluir = set(opciones.get("incluir", ()))
    disp = disponibilidad(ss, firma_actual)
    secciones = []          # (titulo, bloques)
    for _, items in GRUPOS:
        for clave, etiqueta in items:
            if clave not in incluir or not disp[clave]["disponible"]:
                continue
            desact = disp[clave]["desactualizado"]
            if clave == "dataset":
                b = _b_dataset(ss)
            elif clave == "pca":
                b = _b_pca(ss, desact)
            elif clave == "outliers":
                b = _b_outliers(ss, desact)
            elif clave == "dendro":
                b = _b_dendro(ss, desact)
            elif clave.startswith("otros_"):
                b = _b_otros(ss, clave, desact)
            elif clave == "clf":
                b = _b_clf(ss, desact, opciones.get("clf_detalle", []))
            elif clave == "simca":
                b = _b_simca(ss)
            elif clave == "reg":
                b = _b_reg(ss, desact, opciones.get("reg_detalle", []))
            else:
                b = _b_pred(ss)
            titulo_sec = etiqueta + ("  (OUT OF DATE)" if desact else "")
            secciones.append((titulo_sec, b))

    fichas = [n for n in opciones.get("fichas", []) if ss.get("modelos_guardados", {}).get(n, {}).get("ficha")]
    if fichas:
        bf = []
        for n in fichas:
            bf.append({"tipo": "subtitulo", "texto": f"Saved model: {n}"})
            bf.append({"tipo": "ficha", "ficha": ss["modelos_guardados"][n]["ficha"]})
        secciones.append(("Model cards (traceability)", bf))
    if (opciones.get("notas") or "").strip():
        secciones.append(("Notes", [{"tipo": "parrafo", "texto": opciones["notas"].strip()}]))

    indice = [t for t, _ in secciones]
    bloques = [{"tipo": "titulo", "texto": "Contents"},
               {"tipo": "parrafo", "texto": "\n".join(f"{i + 1}. {t}" for i, t in enumerate(indice))}]
    for i, (t, b) in enumerate(secciones):
        bloques.append({"tipo": "salto_pagina"})
        bloques.append({"tipo": "titulo", "texto": f"{i + 1}. {t}"})
        bloques += b
    return (opciones.get("titulo") or "Chemometric analysis report",
            "Generated with Espectrometrika", bloques, indice)
