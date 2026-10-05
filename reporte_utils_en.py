"""
reporte_utils.py
Generador de reportes PDF para la app de quimiometría. Usa matplotlib para
todas las figuras (liviano, sin depender de un navegador/Chrome como
requiere Kaleido) y fpdf2 para el armado del documento.
"""

import io
import re
import datetime
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


def ahora_str(zona_horaria=None, con_segundos=False):
    """Fecha y hora actuales como texto, p.ej. "2026-10-05 14:32 (UTC-03:00)".
    'zona_horaria' es un nombre IANA (p.ej. "America/Argentina/Buenos_Aires",
    la del navegador del usuario). Si falta o es invalida, usa UTC, y lo dice
    explicitamente para que la hora nunca sea ambigua."""
    ahora = None
    if zona_horaria:
        try:
            from zoneinfo import ZoneInfo
            ahora = datetime.datetime.now(ZoneInfo(zona_horaria))
        except Exception:
            ahora = None
    if ahora is None:
        ahora = datetime.datetime.now(datetime.timezone.utc)
    off = ahora.utcoffset() or datetime.timedelta(0)
    minutos = int(off.total_seconds() // 60)
    signo = "+" if minutos >= 0 else "-"
    minutos = abs(minutos)
    formato = "%Y-%m-%d %H:%M:%S" if con_segundos else "%Y-%m-%d %H:%M"
    return f"{ahora.strftime(formato)} (UTC{signo}{minutos // 60:02d}:{minutos % 60:02d})"


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
    fecha_hora = None  # se asigna en crear_reporte()

    def header(self):
        pass

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        izquierda = "Espectrometrika"
        if self.fecha_hora:
            izquierda += f"  |  Report generated {self.fecha_hora}"
        self.cell(0, 10, _sanear(izquierda), align="L")
        self.set_x(self.l_margin)
        self.cell(0, 10, f"Page {self.page_no()}", align="R")


def crear_reporte(fecha_hora=None):
    pdf = ReportePDF(format="A4", unit="mm")
    pdf.fecha_hora = fecha_hora or ahora_str()
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
    if getattr(pdf, "fecha_hora", None):
        pdf.ln(8)
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(*GRIS)
        pdf.multi_cell(0, 6, _sanear(f"Generated on {pdf.fecha_hora}"), align="C")
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


def imagen_desde_png(pdf, png_bytes, ancho_mm=170):
    """Inserta una imagen PNG ya renderizada (p.ej. un clustermap guardado)."""
    pdf.image(io.BytesIO(png_bytes), x=(210 - ancho_mm) / 2, w=ancho_mm)
    pdf.ln(2)


_RE_DECIMALES_LARGOS = re.compile(r"(?<![\w.])(-?\d+)\.(\d{5,})(?:[eE]([+-]?\d+))?(?![\w])")
_RE_NUMERICO = re.compile(r"^[\s<>~+\-]*\d[\d.,]*(?:[eE][+-]?\d+)?\s*%?$")


def _formatear_numeros_largos(texto):
    """Redondea cualquier numero con 5+ decimales dentro de un texto a 4
    decimales (o a 3 cifras significativas en notacion cientifica si es
    diminuto), p.ej. '0.8333333333333334' -> '0.8333'. Los numeros que ya
    vienen con pocos decimales (p.ej. '4030.15') no se tocan."""
    def _redondear(m):
        try:
            valor = float(m.group(0))
        except ValueError:
            return m.group(0)
        if valor != 0 and abs(valor) < 1e-3:
            return f"{valor:.3e}"
        return f"{valor:.4f}"
    return _RE_DECIMALES_LARGOS.sub(_redondear, texto)


def _es_numerico(texto):
    t = texto.strip()
    return bool(t) and bool(_RE_NUMERICO.match(t))


def _preparar_celda(c):
    """Texto listo para el PDF: saneado y con los numeros largos redondeados."""
    if c is None:
        return ""
    return _sanear(_formatear_numeros_largos(str(c)))


def _lineas_celda(pdf, texto, ancho_texto, alto_linea):
    if not texto:
        return 1
    return max(1, len(pdf.multi_cell(max(ancho_texto, 1), alto_linea, texto, split_only=True)))


def _calcular_anchos(pdf, encabezados, filas, ancho_total, tam_fuente, pad):
    """Ancho de cada columna segun su contenido real: ninguna palabra/numero
    queda cortado a la mitad si hay forma de evitarlo."""
    n = len(encabezados)
    pref, minimo = [], []
    for j in range(n):
        pdf.set_font("Helvetica", "B", tam_fuente)
        palabras_h = [w for h in [encabezados[j]] for w in h.split()] or [""]
        min_h = max(pdf.get_string_width(w) for w in palabras_h)
        pref_h = pdf.get_string_width(encabezados[j])
        pdf.set_font("Helvetica", "", tam_fuente)
        min_c, pref_c = 0.0, 0.0
        for fila in filas:
            t = fila[j]
            pref_c = max(pref_c, pdf.get_string_width(t))
            for w in (t.split() or [""]):
                min_c = max(min_c, pdf.get_string_width(w))
        # Un encabezado largo puede partirse en varias lineas (por eso solo
        # su palabra mas larga cuenta para el minimo), pero el contenido de
        # la columna manda para el ancho preferido.
        minimo.append(max(min_h, min_c) + 2 * pad + 0.6)
        pref.append(min(max(pref_c, min_h) + 2 * pad + 0.6, 95.0))
    return pref, minimo


def _repartir_anchos(pref, minimo, ancho_total):
    if sum(minimo) > ancho_total:
        return None  # ni siquiera entran los minimos: hay que achicar la letra
    if sum(pref) <= ancho_total:
        extra = ancho_total - sum(pref)
        # el espacio sobrante se reparte proporcional al ancho preferido
        return [p + extra * p / sum(pref) for p in pref]
    # hay que comprimir: partimos de los minimos y repartimos el resto
    # segun cuanto "quiere" crecer cada columna.
    ganas = [max(p - m, 0.0) for p, m in zip(pref, minimo)]
    libre = ancho_total - sum(minimo)
    total_ganas = sum(ganas) or 1.0
    return [m + libre * g / total_ganas for m, g in zip(minimo, ganas)]


_RE_NUMERO_PURO = re.compile(r"^(-?\d+)(?:\.(\d+))?$")


def _igualar_decimales(filas, n):
    """En cada columna numerica, deja a todos los numeros con la misma
    cantidad de decimales (0.6 -> 0.600 junto a 0.767), asi se leen parejos."""
    for j in range(n):
        celdas = [f[j].strip() for f in filas if f[j].strip()]
        if not celdas:
            continue
        puros = [c for c in celdas if _RE_NUMERO_PURO.match(c)]
        if len(puros) / len(celdas) < 0.7:
            continue
        max_dec = max(len(_RE_NUMERO_PURO.match(c).group(2) or "") for c in puros)
        if max_dec == 0 or max_dec > 4:
            continue
        for f in filas:
            c = f[j].strip()
            m = _RE_NUMERO_PURO.match(c)
            if m:
                dec = len(m.group(2) or "")
                if dec < max_dec:
                    f[j] = (c if "." in c else c + ".") + "0" * (max_dec - dec)
    return filas


def _alineaciones(filas, n):
    res = []
    for j in range(n):
        celdas = [f[j] for f in filas if f[j].strip()]
        numericas = sum(_es_numerico(c) for c in celdas)
        res.append("C" if celdas and numericas / len(celdas) >= 0.7 else "L")
    return res


def _dibujar_tabla(pdf, encabezados, filas, anchos_mm, tam_fuente, pad):
    """Dibuja UNA tabla ya dimensionada. Maneja saltos de pagina (la fila
    nunca se corta y el encabezado se repite) y encabezados multilinea."""
    n = len(encabezados)
    alineaciones = _alineaciones(filas, n)
    alto_linea = max(3.6, tam_fuente * 0.52)
    margen_inferior = pdf.h - pdf.b_margin
    x_inicio_tabla = pdf.l_margin

    def alto_encabezado():
        pdf.set_font("Helvetica", "B", tam_fuente)
        return max(_lineas_celda(pdf, h, w - 2 * pad, alto_linea)
                   for h, w in zip(encabezados, anchos_mm)) * alto_linea + 2.4

    def dibujar_encabezado():
        alto = alto_encabezado()
        pdf.set_font("Helvetica", "B", tam_fuente)
        pdf.set_fill_color(*VERDE)
        pdf.set_text_color(255, 255, 255)
        y = pdf.get_y()
        x = x_inicio_tabla
        for h, w, al in zip(encabezados, anchos_mm, alineaciones):
            pdf.rect(x, y, w, alto, style="F")
            pdf.set_xy(x + pad, y + 1.2)
            pdf.multi_cell(w - 2 * pad, alto_linea, h, border=0, align=al,
                           new_x="RIGHT", new_y="TOP")
            x += w
        pdf.set_xy(x_inicio_tabla, y + alto)
        pdf.set_text_color(0, 0, 0)

    pdf.set_x(x_inicio_tabla)
    # que el encabezado no quede huerfano al pie de la pagina
    if pdf.get_y() + alto_encabezado() + alto_linea * 2 + 2.4 > margen_inferior:
        pdf.add_page()
    dibujar_encabezado()

    for i, fila in enumerate(filas):
        pdf.set_font("Helvetica", "", tam_fuente)
        alto_fila = max(_lineas_celda(pdf, t, w - 2 * pad, alto_linea)
                        for t, w in zip(fila, anchos_mm)) * alto_linea + 2.4

        if pdf.get_y() + alto_fila > margen_inferior:
            pdf.add_page()
            dibujar_encabezado()
            pdf.set_font("Helvetica", "", tam_fuente)

        pdf.set_fill_color(*(GRIS_CLARO if i % 2 else (255, 255, 255)))
        y = pdf.get_y()
        x = x_inicio_tabla
        for t, w, al in zip(fila, anchos_mm, alineaciones):
            pdf.rect(x, y, w, alto_fila, style="F")
            pdf.set_xy(x + pad, y + 1.2)
            pdf.multi_cell(w - 2 * pad, alto_linea, t, border=0, align=al,
                           new_x="RIGHT", new_y="TOP")
            x += w
        pdf.set_draw_color(215, 220, 218)       # linea fina separadora
        pdf.set_line_width(0.15)
        pdf.line(x_inicio_tabla, y + alto_fila, x_inicio_tabla + sum(anchos_mm), y + alto_fila)
        pdf.set_xy(x_inicio_tabla, y + alto_fila)
    pdf.set_font("Helvetica", "", 10.5)
    pdf.ln(3)


def tabla(pdf, encabezados, filas, anchos=None):
    """Tabla para el PDF, pensada para que NUNCA queden numeros superpuestos
    ni cortados a la mitad:

    - Los numeros largos (p.ej. 0.8333333333333334) se redondean a 4 decimales.
    - Los anchos se calculan segun el contenido real de cada columna (o se
      usan los porcentajes de 'anchos' si el llamador los dio y alcanzan).
    - Si no entra a una letra legible (>= 7.5 pt), la tabla se DIVIDE en
      bloques de columnas repitiendo la primera columna (p.ej. el nombre del
      modelo) en cada bloque, en vez de apretar todo ilegiblemente.
    - Encabezados largos en varias lineas; columnas numericas centradas.
    - Ninguna fila queda partida entre dos paginas; el encabezado se repite.
    """
    encabezados = [_preparar_celda(h) for h in encabezados]
    filas = [[_preparar_celda(c) for c in fila] for fila in filas]
    n = len(encabezados)
    if n == 0:
        return
    for fila in filas:               # filas mas cortas que el encabezado
        while len(fila) < n:
            fila.append("")
    filas = _igualar_decimales(filas, n)

    ancho_total = 210 - 2 * 18
    pad = 1.6

    # 1) anchos dados por el llamador, si alcanzan para el contenido
    if anchos is not None and len(anchos) == n:
        candidatos = [ancho_total * a / 100 for a in anchos]
        _, minimo = _calcular_anchos(pdf, encabezados, filas, ancho_total, 9.0, pad)
        if all(c >= m - 0.01 for c, m in zip(candidatos, minimo)):
            _dibujar_tabla(pdf, encabezados, filas, candidatos, 9.0, pad)
            return

    # 2) una sola tabla, con la letra mas grande que permita entrar entera
    for tam in (9.0, 8.5, 8.0, 7.5):
        pref, minimo = _calcular_anchos(pdf, encabezados, filas, ancho_total, tam, pad)
        anchos_mm = _repartir_anchos(pref, minimo, ancho_total)
        if anchos_mm is not None:
            _dibujar_tabla(pdf, encabezados, filas, anchos_mm, tam, pad)
            return

    # 3) demasiado ancha: dividir en bloques de columnas (la 1ra se repite)
    tam = 8.0
    pref, minimo = _calcular_anchos(pdf, encabezados, filas, ancho_total, tam, pad)

    def agrupar(limite):
        grupos, resto = [], list(range(1, n))
        while resto:
            grupo, suma = [0], minimo[0]
            while resto and suma + minimo[resto[0]] <= limite:
                j = resto.pop(0)
                grupo.append(j)
                suma += minimo[j]
            if len(grupo) == 1:      # ni una columna extra entra: forzamos 1
                grupo.append(resto.pop(0))
            grupos.append(grupo)
        return grupos

    grupos = agrupar(ancho_total)
    if len(grupos) > 1:              # equilibrar: 8+7 en vez de 11+4
        limite = minimo[0] + sum(minimo[1:]) / len(grupos) * 1.15
        grupos = agrupar(min(max(limite, minimo[0] + max(minimo[1:])), ancho_total))

    for g, grupo in enumerate(grupos):
        if g > 0:
            if pdf.get_y() + 38 > pdf.h - pdf.b_margin:   # cartel + encabezado + 2 filas
                pdf.add_page()
            pdf.set_font("Helvetica", "I", 8)
            pdf.set_text_color(*GRIS)
            pdf.cell(0, 4.5, "(table continued - remaining columns)", new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(0, 0, 0)
        enc_g = [encabezados[j] for j in grupo]
        filas_g = [[f[j] for j in grupo] for f in filas]
        p_g, m_g = _calcular_anchos(pdf, enc_g, filas_g, ancho_total, tam, pad)
        a_g = _repartir_anchos(p_g, m_g, ancho_total)
        t_g = tam
        if a_g is None:              # ultimo recurso: letra mas chica
            for t_g in (7.5, 7.0, 6.5, 6.0):
                p_g, m_g = _calcular_anchos(pdf, enc_g, filas_g, ancho_total, t_g, pad)
                a_g = _repartir_anchos(p_g, m_g, ancho_total)
                if a_g is not None:
                    break
            if a_g is None:
                total_min = sum(m_g)
                a_g = [m * ancho_total / total_min for m in m_g]
        _dibujar_tabla(pdf, enc_g, filas_g, a_g, t_g, pad)


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


def generar_pdf_ficha_modelo(ficha, fig_extra=None, titulo_fig_extra=None, fecha_hora=None):
    """Arma un PDF autocontenido solo con la ficha de un modelo (para el
    botón de descarga individual). Devuelve el objeto ReportePDF."""
    pdf = crear_reporte(fecha_hora=fecha_hora)
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

def generar_reporte(titulo, subtitulo, secciones, fecha_hora=None):
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
    pdf = crear_reporte(fecha_hora=fecha_hora)
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
        elif tipo == "imagen_png":
            imagen_desde_png(pdf, bloque["png"], ancho_mm=bloque.get("ancho_mm", 170))
        elif tipo == "ficha":
            seccion_ficha_modelo(pdf, bloque["ficha"])
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
