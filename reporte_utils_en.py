"""
reporte_utils.py
Generador de reportes PDF para la app de quimiometría. Usa matplotlib para
todas las figuras (liviano, sin depender de un navegador/Chrome como
requiere Kaleido) y fpdf2 para el armado del documento.
"""

import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fpdf import FPDF

VERDE = (15, 110, 86)
GRIS = (95, 94, 90)
GRIS_CLARO = (240, 245, 243)


_REEMPLAZOS_PDF = {
    "\u2192": "->", "\u2190": "<-", "\u2194": "<->",  # flechas
    "\u00b2": "2", "\u00b3": "3",                      # superíndices ² ³
    "\u00b1": "+/-", "\u00d7": "x", "\u00f7": "/",
    "\u2014": "-", "\u2013": "-", "\u2026": "...",
    "\u00b0": " grados ", "\u00b5": "u",
    "\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'",
    "\u2264": "<=", "\u2265": ">=", "\u2260": "!=",
}


def _sanear(texto):
    """Reemplaza caracteres Unicode comunes que la fuente base de fpdf2 (Helvetica,
    solo Latin-1) no soporta, y descarta cualquier otro carácter no representable
    en vez de romper la generación del PDF."""
    texto = str(texto)
    for original, reemplazo in _REEMPLAZOS_PDF.items():
        texto = texto.replace(original, reemplazo)
    return texto.encode("latin-1", errors="replace").decode("latin-1")


def fig_a_png_bytes(fig, dpi=150):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


class ReportePDF(FPDF):
    def header(self):
        pass

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def crear_reporte():
    pdf = ReportePDF(format="A4", unit="mm")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(18, 16, 18)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(*VERDE)
    return pdf


def titulo_portada(pdf, titulo, subtitulo=None):
    pdf.set_y(70)
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(*VERDE)
    pdf.multi_cell(0, 10, _sanear(titulo), align="C")
    if subtitulo:
        pdf.ln(4)
        pdf.set_font("Helvetica", "", 12)
        pdf.set_text_color(*GRIS)
        pdf.multi_cell(0, 7, _sanear(subtitulo), align="C")
    pdf.set_text_color(0, 0, 0)


def titulo_seccion(pdf, texto):
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(*VERDE)
    pdf.cell(0, 9, _sanear(texto), new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(1)


def subtitulo_seccion(pdf, texto):
    pdf.set_font("Helvetica", "B", 11.5)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(0, 7, _sanear(texto), new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)


def parrafo(pdf, texto):
    pdf.set_font("Helvetica", "", 10.5)
    pdf.multi_cell(0, 5.5, _sanear(texto))
    pdf.ln(1)


def lista_clave_valor(pdf, pares):
    pdf.set_font("Helvetica", "", 10.5)
    for clave, valor in pares:
        pdf.set_font("Helvetica", "B", 10.5)
        ancho_clave = pdf.get_string_width(_sanear(f"{clave}: ")) + 2
        pdf.cell(ancho_clave, 5.8, _sanear(f"{clave}: "))
        pdf.set_font("Helvetica", "", 10.5)
        pdf.multi_cell(0, 5.8, _sanear(valor), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)


def imagen_desde_fig(pdf, fig, ancho_mm=170):
    buf = fig_a_png_bytes(fig)
    y_antes = pdf.get_y()
    pdf.image(buf, x=(210 - ancho_mm) / 2, w=ancho_mm)
    pdf.ln(2)


def _dibujar_encabezado_tabla(pdf, encabezados, anchos):
    pdf.set_font("Helvetica", "B", 9.5)
    pdf.set_fill_color(*VERDE)
    pdf.set_text_color(255, 255, 255)
    for h, w in zip(encabezados, anchos):
        pdf.cell(w, 7, _sanear(h), border=0, fill=True)
    pdf.ln(7)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "", 9)


def tabla(pdf, encabezados, filas, anchos=None):
    """Tabla con salto de página manual: antes de dibujar cada fila se chequea
    si entra en el espacio restante; si no entra, se agrega una página nueva
    (repitiendo el encabezado) ANTES de dibujarla, para que ninguna fila quede
    cortada a la mitad entre dos páginas."""
    ancho_total = 210 - 2 * 18
    if anchos is None:
        anchos = [ancho_total / len(encabezados)] * len(encabezados)
    else:
        anchos = [ancho_total * a / 100 for a in anchos]

    margen_inferior = pdf.h - pdf.b_margin

    _dibujar_encabezado_tabla(pdf, encabezados, anchos)

    for i, fila in enumerate(filas):
        textos = [_sanear(c) for c in fila]
        alturas = []
        for texto, w in zip(textos, anchos):
            n_lineas = max(1, len(pdf.multi_cell(w, 5, texto, split_only=True)))
            alturas.append(n_lineas * 5)
        alto_fila = max(alturas)

        if pdf.get_y() + alto_fila > margen_inferior:
            pdf.add_page()
            _dibujar_encabezado_tabla(pdf, encabezados, anchos)

        pdf.set_fill_color(*(GRIS_CLARO if i % 2 else (255, 255, 255)))
        x_inicio = pdf.get_x()
        y_inicio = pdf.get_y()
        for texto, w in zip(textos, anchos):
            x = pdf.get_x()
            y = pdf.get_y()
            pdf.rect(x, y, w, alto_fila, style="F")
            pdf.set_xy(x, y)
            pdf.multi_cell(w, 5, texto, border=0, align="L", new_x="RIGHT", new_y="TOP")
        pdf.set_xy(x_inicio, y_inicio + alto_fila)
    pdf.ln(2)


def salto_pagina(pdf):
    pdf.add_page()


# =============================================================================
# FIGURAS MATPLOTLIB REUTILIZABLES PARA EL REPORTE
# =============================================================================

def fig_espectros(numeros_onda, X, clases=None, titulo="Spectra"):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    if clases is not None:
        clases_unicas = np.unique(clases)
        colores = plt.cm.tab10(np.linspace(0, 1, len(clases_unicas)))
        mapa_color = dict(zip(clases_unicas, colores))
        for c in clases_unicas:
            mask = clases == c
            for i, idx in enumerate(np.where(mask)[0]):
                ax.plot(numeros_onda, X[idx], color=mapa_color[c], alpha=0.6, linewidth=0.7,
                        label=str(c) if i == 0 else None)
        ax.legend(fontsize=8, frameon=False)
    else:
        for i in range(X.shape[0]):
            ax.plot(numeros_onda, X[i], color="steelblue", alpha=0.5, linewidth=0.7)
    ax.set_xlabel("Wavenumber")
    ax.set_ylabel("Signal")
    ax.set_title(titulo, fontsize=11)
    if numeros_onda[0] > numeros_onda[-1]:
        ax.invert_xaxis()
    fig.tight_layout()
    return fig


def fig_variables_seleccionadas(numeros_onda, espectro_promedio, mascara, titulo="Selected variables"):
    """Mean spectrum with the regions selected by Boruta/GA highlighted."""
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(numeros_onda, espectro_promedio, color="#5F5E5A", linewidth=1.2, label="Mean spectrum")
    if mascara is not None and mascara.any():
        idx_sel = np.where(mascara)[0]
        # Agrupar índices consecutivos en bloques, para pintar franjas en vez de líneas sueltas
        bloques = np.split(idx_sel, np.where(np.diff(idx_sel) != 1)[0] + 1)
        for j, bloque in enumerate(bloques):
            x0 = numeros_onda[bloque[0]]
            x1 = numeros_onda[bloque[-1]]
            ax.axvspan(min(x0, x1), max(x0, x1), color="#0F6E56", alpha=0.25,
                       label="Selected variables" if j == 0 else None)
    ax.set_xlabel("Wavenumber")
    ax.set_ylabel("Signal")
    ax.set_title(titulo, fontsize=11)
    ax.legend(fontsize=8, frameon=False)
    if numeros_onda[0] > numeros_onda[-1]:
        ax.invert_xaxis()
    fig.tight_layout()
    return fig


def fig_scree(var_explicada, n_comp_marcado=None):
    fig, ax1 = plt.subplots(figsize=(7, 3))
    var_acumulada = np.cumsum(var_explicada)
    n = min(15, len(var_explicada))
    ax1.bar(range(1, n + 1), var_explicada[:n], color="#185FA5", alpha=0.7)
    ax1.set_xlabel("Principal component")
    ax1.set_ylabel("% individual")
    ax2 = ax1.twinx()
    ax2.plot(range(1, n + 1), var_acumulada[:n], color="#B91C1C", marker="o", markersize=3)
    ax2.set_ylabel("% acumulado")
    if n_comp_marcado:
        ax1.axvline(n_comp_marcado, color="gray", linestyle="--", linewidth=1)
    fig.tight_layout()
    return fig


def fig_scores(scores, var_explicada, pc_x, pc_y, clases=None, ids=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    if clases is not None:
        clases_unicas = np.unique(clases)
        colores = plt.cm.tab10(np.linspace(0, 1, len(clases_unicas)))
        for c, color in zip(clases_unicas, colores):
            mask = clases == c
            ax.scatter(scores[mask, pc_x - 1], scores[mask, pc_y - 1], color=color, s=28, alpha=0.8, label=str(c))
        ax.legend(fontsize=8, frameon=False)
    else:
        ax.scatter(scores[:, pc_x - 1], scores[:, pc_y - 1], color="steelblue", s=28, alpha=0.8)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel(f"PC{pc_x} ({var_explicada[pc_x-1]:.1f}%)")
    ax.set_ylabel(f"PC{pc_y} ({var_explicada[pc_y-1]:.1f}%)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def fig_outliers(T2, Q, T2_lim, Q_lim, T2_lo, T2_hi, Q_lo, Q_hi, clases=None):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    if clases is not None:
        clases_unicas = np.unique(clases)
        colores = plt.cm.tab10(np.linspace(0, 1, len(clases_unicas)))
        for c, color in zip(clases_unicas, colores):
            mask = clases == c
            ax.scatter(T2[mask], Q[mask], color=color, s=26, alpha=0.8, label=str(c))
        ax.legend(fontsize=8, frameon=False)
    else:
        ax.scatter(T2, Q, color="#5F5E5A", s=26, alpha=0.8)
    ax.axvline(T2_lim, color="#B91C1C", linestyle="--", linewidth=1.2)
    ax.axhline(Q_lim, color="#B91C1C", linestyle="--", linewidth=1.2)
    ax.set_xlim(T2_lo, T2_hi)
    ax.set_ylim(Q_lo, Q_hi)
    ax.set_xlabel("Hotelling's T²")
    ax.set_ylabel("Q residual")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def fig_matriz_confusion(matriz, clases):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(matriz, cmap="Greens")
    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            color = "white" if matriz[i, j] > matriz.max() * 0.6 else "#333"
            ax.text(j, i, matriz[i, j], ha="center", va="center", color=color, fontsize=11, weight="bold")
    ax.set_xticks(range(len(clases))); ax.set_xticklabels(clases)
    ax.set_yticks(range(len(clases))); ax.set_yticklabels(clases)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    fig.tight_layout()
    return fig


def fig_predicho_vs_real(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(y_true, y_pred, color="#185FA5", s=26, alpha=0.75)
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], color="gray", linestyle="--", linewidth=1.2)
    ax.set_xlabel("True value")
    ax.set_ylabel("Predicted value")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def fig_roc_curve(y_true, y_proba, clases):
    """Curva ROC uno-contra-el-resto por cada clase, con su AUC en la leyenda."""
    from sklearn.metrics import roc_curve, auc as auc_sklearn
    from sklearn.preprocessing import label_binarize

    y_true = np.asarray(y_true)
    fig, ax = plt.subplots(figsize=(5.5, 5))

    if len(clases) == 2:
        y_bin = (y_true == clases[1]).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, y_proba[:, 1])
        ax.plot(fpr, tpr, color="#0F6E56", linewidth=1.8,
                label=f"AUC = {auc_sklearn(fpr, tpr):.3f}")
    else:
        y_bin = label_binarize(y_true, classes=clases)
        colores = plt.cm.tab10(np.linspace(0, 1, len(clases)))
        for i, (c, color) in enumerate(zip(clases, colores)):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            ax.plot(fpr, tpr, color=color, linewidth=1.5,
                    label=f"{c} (AUC={auc_sklearn(fpr, tpr):.3f})")

    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


# =============================================================================
# FICHA DEL MODELO (trazabilidad / accountability)
# =============================================================================

def seccion_ficha_modelo(pdf, ficha):
    """
    Renderiza la 'ficha del modelo': metadata de trazabilidad (cuándo se
    entrenó, con qué datos, qué pretratamiento, qué hiperparámetros y
    métricas dio, y con qué versiones de software). 'ficha' es el diccionario
    guardado junto con el modelo (ver construir_ficha_modelo en app.py).
    """
    titulo_seccion(pdf, "Model card (traceability)")

    lista_clave_valor(pdf, [
        ("Traceability ID", ficha.get("id_trazabilidad", "n/a")),
        ("Training date and time", ficha.get("fecha_creacion", "n/a")),
        ("Model name", ficha.get("nombre_modelo", "n/a")),
        ("Task type", ficha.get("tipo", "n/a")),
        ("Algorithm", ficha.get("algoritmo", "n/a")),
    ])

    subtitulo_seccion(pdf, "Input data")
    lista_clave_valor(pdf, [
        ("Number of samples used", ficha.get("n_muestras", "n/a")),
        ("Number of original variables", ficha.get("n_variables_totales", "n/a")),
        ("Variables used by the model", ficha.get("n_variables_usadas", "n/a")),
        ("Class distribution / Y variable", ficha.get("descripcion_y", "n/a")),
    ])

    subtitulo_seccion(pdf, "Preprocessing and variable selection")
    lista_clave_valor(pdf, [
        ("Preprocessing applied", ficha.get("pretratamiento_desc", "none")),
        ("Variable selection method", ficha.get("seleccion_variables_desc", "none")),
    ])

    if ficha.get("hiperparametros"):
        subtitulo_seccion(pdf, "Optimized hyperparameters")
        lista_clave_valor(pdf, [("Optimization method", ficha.get("metodo_optimizacion", "n/a"))]
                           + [(k, v) for k, v in ficha["hiperparametros"].items()])

    subtitulo_seccion(pdf, "Validation configuration")
    lista_clave_valor(pdf, [
        ("Cross-validation", ficha.get("cv_desc", "n/a")),
        ("Independent test proportion", ficha.get("prop_test_desc", "n/a")),
    ])

    if ficha.get("metricas"):
        subtitulo_seccion(pdf, "Performance metrics")
        filas = [[k, str(v)] for k, v in ficha["metricas"].items()]
        tabla(pdf, ["Metric", "Value"], filas, anchos=[40, 60])

    subtitulo_seccion(pdf, "Software environment")
    entorno = ficha.get("entorno_software", {})
    lista_clave_valor(pdf, [(k, v) for k, v in entorno.items()])


def generar_pdf_ficha_modelo(ficha, fig_extra=None, titulo_fig_extra=None):
    """Arma un PDF autocontenido solo con la ficha de un modelo (para el
    botón de descarga individual). Devuelve el objeto ReportePDF."""
    pdf = crear_reporte()
    titulo_portada(pdf, "Model card", f"ID: {ficha.get('id_trazabilidad', 'n/a')}")
    salto_pagina(pdf)
    seccion_ficha_modelo(pdf, ficha)
    if fig_extra is not None:
        if titulo_fig_extra:
            subtitulo_seccion(pdf, titulo_fig_extra)
        imagen_desde_fig(pdf, fig_extra)
    return pdf


# =============================================================================
# GENERADOR GENÉRICO DE REPORTES (por secciones)
# =============================================================================

def generar_reporte(titulo, subtitulo, secciones):
    """
    Arma un PDF a partir de una lista de "bloques" descriptivos, para no
    tener que repetir el armado del documento en cada pestaña de la app.

    Cada bloque de 'secciones' es un dict con "tipo" y sus datos:
      {"tipo": "titulo", "texto": "..."}
      {"tipo": "subtitulo", "texto": "..."}
      {"tipo": "parrafo", "texto": "..."}
      {"tipo": "clave_valor", "pares": [(clave, valor), ...]}
      {"tipo": "tabla", "encabezados": [...], "filas": [[...], ...], "anchos": [...] (opcional)}
      {"tipo": "imagen", "fig": <figura matplotlib>, "ancho_mm": 170 (opcional)}
      {"tipo": "salto_pagina"}
      {"tipo": "espacio"}
    """
    pdf = crear_reporte()
    titulo_portada(pdf, titulo, subtitulo)
    salto_pagina(pdf)

    for bloque in secciones:
        tipo = bloque["tipo"]
        if tipo == "titulo":
            titulo_seccion(pdf, bloque["texto"])
        elif tipo == "subtitulo":
            subtitulo_seccion(pdf, bloque["texto"])
        elif tipo == "parrafo":
            parrafo(pdf, bloque["texto"])
        elif tipo == "clave_valor":
            lista_clave_valor(pdf, bloque["pares"])
        elif tipo == "tabla":
            tabla(pdf, bloque["encabezados"], bloque["filas"], anchos=bloque.get("anchos"))
        elif tipo == "imagen":
            imagen_desde_fig(pdf, bloque["fig"], ancho_mm=bloque.get("ancho_mm", 170))
        elif tipo == "salto_pagina":
            salto_pagina(pdf)
        elif tipo == "espacio":
            pdf.ln(4)
    return pdf


# =============================================================================
# LEARNING CURVE AND MODEL-COMPARISON FIGURES
# =============================================================================

def fig_curva_aprendizaje(curva, nombre_modelo=""):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ts = curva["train_sizes"]
    ax.plot(ts, curva["train_scores_mean"], color="#185FA5", marker="o", markersize=4, label="Training score")
    ax.fill_between(ts, curva["train_scores_mean"] - curva["train_scores_std"],
                     curva["train_scores_mean"] + curva["train_scores_std"], color="#185FA5", alpha=0.15)
    ax.plot(ts, curva["val_scores_mean"], color="#B91C1C", marker="o", markersize=4, label="Cross-validation score")
    ax.fill_between(ts, curva["val_scores_mean"] - curva["val_scores_std"],
                     curva["val_scores_mean"] + curva["val_scores_std"], color="#B91C1C", alpha=0.15)
    ax.set_xlabel("Training set size (samples)")
    ax.set_ylabel(curva.get("scoring", "score"))
    ax.set_title(f"Learning curve — {nombre_modelo}" if nombre_modelo else "Learning curve", fontsize=11)
    ax.legend(fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def fig_comparacion_pvalores(pvalores_df):
    """Heatmap of pairwise p-values between models (paired t-test on
    per-fold CV scores). Values below 0.05 are highlighted."""
    fig, ax = plt.subplots(figsize=(max(4.5, 0.9 * len(pvalores_df)), max(4, 0.9 * len(pvalores_df))))
    datos = pvalores_df.values
    im = ax.imshow(datos, cmap="RdYlGn", vmin=0, vmax=1)
    for i in range(datos.shape[0]):
        for j in range(datos.shape[1]):
            valor = datos[i, j]
            color = "white" if valor < 0.2 or valor > 0.8 else "#333"
            texto = "-" if i == j else f"{valor:.3f}"
            peso = "bold" if (i != j and valor < 0.05) else "normal"
            ax.text(j, i, texto, ha="center", va="center", color=color, fontsize=9, weight=peso)
    ax.set_xticks(range(len(pvalores_df.columns))); ax.set_xticklabels(pvalores_df.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pvalores_df.index))); ax.set_yticklabels(pvalores_df.index)
    ax.set_title("Pairwise p-values (paired t-test on CV folds)\nbold < 0.05 = likely a real difference", fontsize=9.5)
    fig.tight_layout()
    return fig


def fig_dendrograma(Z, etiquetas, clases=None):
    """Static dendrogram for PDF reports, with optional leaf-label coloring by class."""
    from scipy.cluster import hierarchy
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    dend = hierarchy.dendrogram(Z, labels=list(etiquetas), ax=ax, color_threshold=0.7 * max(Z[:, 2]),
                                 above_threshold_color="#8A8A85")
    ax.set_ylabel("Distance")
    ax.spines[["top", "right"]].set_visible(False)
    if clases is not None:
        clases_unicas = list(dict.fromkeys(clases))
        colores = plt.cm.Set1(np.linspace(0, 1, len(clases_unicas)))
        mapa_color = dict(zip(clases_unicas, colores))
        id_a_clase = dict(zip(etiquetas, clases))
        for tick_label in ax.get_xmajorticklabels():
            clase_muestra = id_a_clase.get(tick_label.get_text())
            if clase_muestra in mapa_color:
                tick_label.set_color(mapa_color[clase_muestra])
                tick_label.set_weight("bold")
        handles = [plt.Line2D([0], [0], marker="s", linestyle="", color=c, label=cl)
                   for cl, c in mapa_color.items()]
        ax.legend(handles=handles, fontsize=8, frameon=False, loc="upper right")
    fig.tight_layout()
    return fig
