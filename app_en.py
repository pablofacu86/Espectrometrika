"""
app_en.py — Espectrometrika
Streamlit application: spectra/chromatogram loading, preprocessing, PCA,
outlier detection, cluster analysis, classification, regression and
prediction on new samples — with a point-and-click interface (no coding
required). Includes model persistence, hyperparameter optimization, and a
downloadable traceability card (model card) for every trained model.

To run it:  streamlit run app_en.py
"""

import io
import os
import base64
import joblib
import smtplib
import datetime
from email.message import EmailMessage
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from scipy.cluster import hierarchy
from sklearn.decomposition import PCA

import chemo_utils as cu
import models_utils_en as mu
import reporte_utils_en as ru

st.set_page_config(
    page_title="Espectrometrika",
    page_icon=os.path.join(os.path.dirname(__file__), "assets", "favicon.png"),
    layout="wide",
)

# Brand palette (kept as module-level constants so other parts of the app can reuse them)
PETROLEUM = "#0B3D54"
PETROLEUM_LIGHT = "#154D6B"
TEAL = "#14B8A6"
TEAL_LIGHT = "#5EEAD4"
TEAL_SOFT = "#E6FBF7"


def _img_b64(nombre_archivo):
    """Reads an image from assets/ and returns it as a base64 data URI, for inline HTML embedding."""
    ruta = os.path.join(os.path.dirname(__file__), "assets", nombre_archivo)
    try:
        with open(ruta, "rb") as f:
            datos = base64.b64encode(f.read()).decode()
        return f"data:image/png;base64,{datos}"
    except FileNotFoundError:
        return None


CONTACTO_DESTINATARIO = "pfsanchez@inti.gob.ar"


def enviar_mensaje_contacto(mensaje, nombre_remitente, email_remitente):
    """Sends a feedback message to the app maintainer's inbox.

    Tries Formspree first (simplest to set up — no email credentials needed,
    just a free account at formspree.io), and falls back to sending via SMTP
    if Formspree isn't configured but SMTP secrets are.

    Expected structure in .streamlit/secrets.toml, EITHER of:

        [formspree]
        endpoint = "https://formspree.io/f/your-form-id"

    OR

        [smtp]
        server = "smtp.your-provider.com"
        port = 587
        user = "your-sending-address@example.com"
        password = "your-app-password"

    Returns (exito: bool, detalle: str).
    """
    cuerpo = mensaje.strip()
    if nombre_remitente.strip():
        cuerpo += f"\n\n— Sent by: {nombre_remitente.strip()}"
    if email_remitente.strip():
        cuerpo += f" ({email_remitente.strip()})"
    cuerpo += f"\n— Sent from Espectrometrika on {datetime.datetime.now():%Y-%m-%d %H:%M}"

    # --- Option 1: Formspree (no email credentials needed) ---
    # Checked in two possible places, since different hosting platforms expose
    # configuration differently:
    #   - Posit Connect Cloud "Secret variables" become plain environment
    #     variables, so we check os.environ first.
    #   - Local development / Streamlit Community Cloud use a
    #     .streamlit/secrets.toml file, read through st.secrets.
    endpoint = os.environ.get("FORMSPREE_ENDPOINT")
    if not endpoint:
        try:
            endpoint = st.secrets["formspree"]["endpoint"]
        except Exception:
            endpoint = None
    if endpoint:
        # Defensive cleanup: strip whitespace and a common copy/paste mistake
        # (pasting a label like "URL: " or "Endpoint: " along with the value).
        endpoint = endpoint.strip()
        for _prefijo in ("URL:", "Url:", "url:", "Endpoint:", "endpoint:"):
            if endpoint.startswith(_prefijo):
                endpoint = endpoint[len(_prefijo):].strip()
        if not endpoint.startswith(("http://", "https://")):
            return False, (f"The configured Formspree endpoint doesn't look like a valid URL "
                            f"({endpoint!r}). Check the FORMSPREE_ENDPOINT value — it should be "
                            f"only the URL, e.g. https://formspree.io/f/xxxxxxxx, nothing else.")
        try:
            import requests
            respuesta = requests.post(
                endpoint,
                data={"message": cuerpo, "name": nombre_remitente.strip(),
                      "_replyto": email_remitente.strip() or "no-reply@espectrometrika.app"},
                headers={"Accept": "application/json"},
                timeout=15,
            )
            if respuesta.status_code in (200, 202):
                return True, "Message sent — thank you!"
            return False, f"Could not send the message right now (Formspree said: {respuesta.status_code})."
        except Exception as e:
            return False, f"Could not send the message right now ({e})."

    # --- Option 2: SMTP fallback ---
    # Same idea: plain env vars first (Posit Connect Cloud), then secrets.toml.
    if os.environ.get("SMTP_SERVER"):
        smtp_cfg = {
            "server": os.environ.get("SMTP_SERVER"),
            "port": os.environ.get("SMTP_PORT", 587),
            "user": os.environ.get("SMTP_USER"),
            "password": os.environ.get("SMTP_PASSWORD"),
        }
    else:
        try:
            smtp_cfg = st.secrets["smtp"]
        except Exception:
            return False, ("Feedback form is not configured yet (missing Formspree or SMTP "
                            "configuration). Ask the app maintainer to set it up.")

    msg = EmailMessage()
    msg["Subject"] = "Espectrometrika — New feedback message"
    msg["From"] = smtp_cfg["user"]
    msg["To"] = CONTACTO_DESTINATARIO
    if email_remitente.strip():
        msg["Reply-To"] = email_remitente.strip()
    msg.set_content(cuerpo)

    try:
        with smtplib.SMTP(smtp_cfg["server"], int(smtp_cfg.get("port", 587))) as servidor:
            servidor.starttls()
            servidor.login(smtp_cfg["user"], smtp_cfg["password"])
            servidor.send_message(msg)
        return True, "Message sent — thank you!"
    except Exception as e:
        return False, f"Could not send the message right now ({e})."

# Custom CSS for a more polished, "scientific software" look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .block-container {padding-top: 3.5rem;}
    h1, h2, h3 {color: #0B3D54; font-family: 'Inter', sans-serif; font-weight: 700;}
    div[data-testid="stMetricValue"] {color: #0B3D54;}
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        flex-wrap: wrap;
        row-gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 16px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        color: #0B3D54 !important;
        border-bottom-color: #14B8A6 !important;
    }
    /* "Home" is always the first tab — give it its own accent color so it
       stands out from the rest, even when it isn't the selected tab. */
    .stTabs [data-baseweb="tab-list"] button:first-child {
        color: #D97706 !important;
    }
    .stTabs [data-baseweb="tab-list"] button:first-child[aria-selected="true"] {
        color: #D97706 !important;
        border-bottom-color: #D97706 !important;
    }
    .stTabs [data-baseweb="tab-list"] button:first-child p {
        color: #D97706 !important;
        font-weight: 700 !important;
    }
    div[data-testid="stSidebar"] {
        border-right: 1px solid rgba(11, 61, 84, 0.12);
    }
    .stButton>button, .stDownloadButton>button {
        border-radius: 8px;
        font-family: 'Inter', sans-serif;
    }
    .stButton>button:hover, .stDownloadButton>button:hover {
        border-color: #14B8A6;
        color: #0B3D54;
    }
    a {color: #14B8A6 !important;}
</style>
""", unsafe_allow_html=True)

# Well-distinguished color palette for coloring by class (avoids Plotly
# falling back to a continuous color scale with just a couple of similar
# shades when there are only 2-3 classes).
CLASS_PALETTE = px.colors.qualitative.Set1


def rgb_string_a_hex(color):
    """Converts a Plotly 'rgb(r,g,b)' color string to '#rrggbb' hex, so the
    same palette can also be used with matplotlib (which doesn't accept the
    'rgb(...)' CSS-style string format). Leaves already-hex colors untouched."""
    if isinstance(color, str) and color.startswith("rgb"):
        r, g, b = [int(v) for v in color[color.index("(") + 1: color.index(")")].split(",")]
        return f"#{r:02x}{g:02x}{b:02x}"
    return color

# Hyperparameter-search speed/quality presets. The tuple is (internal method
# name passed to mu.optimizar_hiperparametros, explanation shown to the user).
PRESETS_OPTIMIZACION = {
    "Fast": ("fast",
             "Tries a small number of random combinations (8). The quickest option — "
             "good for a first pass, may miss the exact optimum."),
    "Balanced": ("balanced",
                 "Successive-halving search from random combinations: quickly discards weak "
                 "options and only spends full effort on the promising ones. Good speed/quality "
                 "trade-off — recommended default."),
    "Thorough": ("thorough",
                 "Successive-halving search over the ENTIRE grid: still checks every "
                 "combination, just far more efficiently than a classic grid search."),
    "Bayesian": ("bayesian",
                 "Each combination it tries is chosen using what it learned from every "
                 "previous one (Optuna/TPE), so the search homes in on promising regions "
                 "instead of sampling blindly. Often finds a better result with fewer trials "
                 "than the other options — a good choice when training a single model."),
}


def resetear_prefijo(prefijo):
    """Clears every session_state key starting with 'prefijo' — used by the
    per-tab 'Reset' buttons so leftover results from a previous run (a
    different set of models, an old learning curve, an old p-value table...)
    never linger on screen after starting a new analysis in that tab."""
    claves_a_borrar = [k for k in st.session_state.keys() if k.startswith(prefijo)]
    for k in claves_a_borrar:
        del st.session_state[k]


def df_a_excel_bytes(hojas):
    """Builds an in-memory .xlsx from a dict {sheet_name: DataFrame}."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for nombre_hoja, df_hoja in hojas.items():
            df_hoja.to_excel(writer, sheet_name=nombre_hoja[:31], index=False)
    return buffer.getvalue()


# =============================================================================
# CACHED, EXPENSIVE COMPUTATIONS
# -----------------------------------------------------------------------
# Streamlit reruns the ENTIRE script top-to-bottom on every single widget
# interaction, anywhere in the app — including tabs that have nothing to do
# with the widget that was actually touched. Without caching, that means
# toggling something as unrelated as "variable selection method" in the
# Classification tab would still silently recompute preprocessing, refit
# PCA, and recompute outlier statistics from scratch every single time,
# which is the main reason simple UI interactions could feel sluggish
# (especially on a shared/limited CPU). These cached wrappers make each of
# those only actually recompute when their real inputs change.
# =============================================================================

@st.cache_data(show_spinner=False)
def _leer_archivo_cacheado(bytes_archivo, nombre_archivo, hoja, fila_encabezado):
    """Cached raw file read + header-row parsing, keyed by the file's own
    bytes (not just its name) — so re-parsing only happens when the file, the
    chosen sheet, or the header row actually change, not on every rerun."""
    buffer = io.BytesIO(bytes_archivo)
    if nombre_archivo.lower().endswith(".csv"):
        return pd.read_csv(buffer, header=fila_encabezado - 1)
    return pd.read_excel(buffer, header=fila_encabezado - 1, sheet_name=hoja)


@st.cache_data(show_spinner=False)
def _leer_preview_cacheada(bytes_archivo, nombre_archivo, hoja):
    """Cached raw preview (first rows, no header assumed) for the
    'raw preview' expander."""
    buffer = io.BytesIO(bytes_archivo)
    if nombre_archivo.lower().endswith(".csv"):
        return pd.read_csv(buffer, header=None, nrows=6)
    return pd.read_excel(buffer, header=None, nrows=6, sheet_name=hoja)


@st.cache_data(show_spinner=False)
def _aplicar_pretratamiento_cacheado(X_paso0, secuencia):
    """Cached preprocessing: only recomputes when the raw data or the chosen
    steps/parameters actually change, not on every unrelated rerun."""
    X_pret = X_paso0.copy()
    for paso_tup in secuencia:
        X_pret = aplicar_paso(X_pret, paso_tup)
    return X_pret


@st.cache_data(show_spinner=False)
def _calcular_linkage_cacheado(X_hca, metodo):
    """Cached hierarchical clustering: hierarchy.linkage does real (O(n^2)-O(n^3))
    work, and with no caching it was re-running on EVERY script rerun — even
    ones triggered by an unrelated widget in a completely different tab, since
    Streamlit re-executes the whole script top to bottom each time. Caching it
    means it only recomputes when the data or the linkage method actually change."""
    return hierarchy.linkage(X_hca, method=metodo, metric="euclidean")


@st.cache_resource(show_spinner=False)
def _ajustar_pca_cacheado(X_pca_input, n_comp_max):
    """Cached PCA fit. Uses cache_resource (not cache_data) because it
    returns the fitted scikit-learn PCA object itself, which other tabs
    reuse directly (e.g. for its .transform() and .mean_) — cache_resource
    avoids deep-copying that object on every cache hit."""
    return PCA(n_components=n_comp_max).fit(X_pca_input)


@st.cache_data(show_spinner=False)
def _calcular_outliers_cacheado(X_pca_input, scores_completo, cargas_completo,
                                 autovalores, media_pca, n_comp, alpha):
    """Cached Hotelling's T² / residual Q computation for the Outliers tab."""
    T2 = cu.calcular_T2(scores_completo, autovalores, n_comp)
    T2_lim = cu.limite_T2(X_pca_input.shape[0], n_comp, alpha)
    Q = cu.calcular_Q(X_pca_input, scores_completo, cargas_completo, n_comp, media=media_pca)
    Q_lim = cu.limite_Q(autovalores, n_comp, alpha)
    Q_lim_confiable = cu.limite_Q_confiable(Q, Q_lim)
    if not Q_lim_confiable:
        Q_lim = float(np.percentile(Q, 100 * (1 - alpha)))
    return T2, T2_lim, Q, Q_lim, Q_lim_confiable


def df_variables_seleccionadas(eje, mascara):
    """Table with the detail of the selected variables (wavenumbers)."""
    idx_sel = np.where(mascara)[0]
    return pd.DataFrame({
        "position": idx_sel,
        "wavenumber": eje[idx_sel],
    })


def como_texto(clases):
    """Forces classes to always be text/categorical (never numeric), so that
    Plotly always colors them as well-differentiated discrete categories,
    instead of interpreting them as a continuous numeric variable (which
    would give a single-hue scale, e.g. two similar shades of blue)."""
    if clases is None:
        return None
    return np.array([str(c) for c in clases])


def ejes_a_float(valores):
    """Converts a list/array of axis labels (wavenumbers, retention times,
    chemical shifts...) to float, auto-fixing the European decimal comma
    (e.g. "1500,5" -> "1500.5") when needed — some instruments/software
    export spectra with a comma instead of a dot as the decimal separator,
    and plain float() can't parse that on its own."""
    valores_texto = [str(v).strip() for v in valores]
    try:
        return np.array(valores_texto, dtype=float)
    except ValueError:
        pass

    def _arreglar(v):
        if "," in v and "." not in v:
            # Only a comma present: it's a decimal separator (e.g. "1500,5").
            return v.replace(",", ".")
        if "," in v and "." in v:
            # Both present: the comma is a thousands separator (e.g. "1,500.5").
            return v.replace(",", "")
        return v

    return np.array([_arreglar(v) for v in valores_texto], dtype=float)


# =============================================================================
# SESSION STATE
# =============================================================================

def init_state():
    defaults = {
        "df": None,
        "ids": None,
        "numeros_onda": None,
        "numeros_onda_pret": None,
        "X": None,
        "clases": None,
        "mascara_excluidas": None,   # bool array: True = sample excluded as outlier
        "modelos_guardados": {},     # trained models saved for the Prediction tab
        "valores_y": None,           # continuous target (y) for Regression
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


def hay_datos():
    return st.session_state.X is not None


def indice_activo():
    """Indices of the samples currently included (not excluded as outliers)."""
    n = st.session_state.X.shape[0]
    if st.session_state.mascara_excluidas is None:
        st.session_state.mascara_excluidas = np.zeros(n, dtype=bool)
    return ~st.session_state.mascara_excluidas


def datos_activos():
    """Returns (ids, X, clases) filtered by the exclusion mask.
    Classes are always returned as text (see como_texto)."""
    idx = indice_activo()
    ids = st.session_state.ids[idx]
    X = st.session_state.X[idx]
    clases = como_texto(st.session_state.clases[idx]) if st.session_state.clases is not None else None
    return ids, X, clases


# =============================================================================
# "COMPUTED WITH..." STATE FINGERPRINT
# -----------------------------------------------------------------------
# PCA, outlier detection, classification and regression are all fairly
# expensive, and every one of them depends on preprocessing + which samples
# are currently excluded as outliers. Recomputing them automatically every
# time ANYTHING changes elsewhere in the app (even in an unrelated tab) is
# wasteful and is the main reason the app can feel slow. Instead, each of
# these results is computed only on an explicit user action (first visit,
# or clicking an "Update" button), and is tagged with a lightweight
# "fingerprint" of the data/preprocessing/outlier state it was computed
# with. If that fingerprint no longer matches the CURRENT state, a small
# badge says so — but the old result stays on screen until the user
# actually asks for an update.
# =============================================================================

def firma_datos_activos():
    """A cheap fingerprint (shape + sum + sum-of-squares, NOT a full hash) of
    the currently active, preprocessed dataset — fast even for large arrays,
    and virtually impossible to collide on by chance. Returns None if there's
    no preprocessed data yet."""
    X_pret = st.session_state.get("X_pret")
    if X_pret is None:
        return None
    idx = indice_activo()
    X_activo = X_pret[idx]
    mascara = st.session_state.get("mascara_excluidas")
    pasos = st.session_state.get("pasos_pretratamiento")
    return (
        X_activo.shape,
        round(float(np.sum(X_activo)), 4),
        round(float(np.sum(X_activo ** 2)), 4),
        int(mascara.sum()) if mascara is not None else 0,
        str(pasos),
    )


def descripcion_firma(firma):
    """Human-readable one-liner for a firma_datos_activos() tuple, for the badge."""
    if firma is None:
        return "no data"
    n_muestras, n_vars = firma[0]
    n_excluidas = firma[3]
    pasos_txt = firma[4]
    resumen_pasos = "no preprocessing" if pasos_txt in ("None", "[]", "") else pasos_txt
    return f"{n_muestras} samples × {n_vars} variables, {n_excluidas} excluded as outliers, preprocessing: {resumen_pasos}"


def insignia_estado(clave_firma, etiqueta="This result"):
    """Compares a stored fingerprint (session_state[clave_firma]) against the
    CURRENT one and renders a small badge. Returns True if the stored result
    is stale (current state has since changed) or doesn't exist yet."""
    firma_guardada = st.session_state.get(clave_firma)
    firma_actual = firma_datos_activos()
    if firma_guardada is None:
        return True
    if firma_guardada == firma_actual:
        st.caption(f"✅ Up to date — computed with: {descripcion_firma(firma_guardada)}.")
        return False
    st.warning(
        f"⚠️ **{etiqueta} is out of date.** It was computed with: "
        f"{descripcion_firma(firma_guardada)}. The current data/preprocessing/outliers "
        f"are different now. The numbers below are still the last ones you asked for — "
        f"click **Update** below whenever you're ready to recompute them.",
        icon="⚠️",
    )
    return True


# Preprocessing functions that are not part of the original
# cu.aplicar_pretratamientos "catalog" (added for Raman/NMR/chromatograms).
# Shared between the Preprocessing tab and the Prediction tab, so the exact
# same steps can be reapplied to new samples.
FUNCIONES_EXTRA = {
    "pqn": cu.pqn,
    "linea_base_als": cu.linea_base_als,
    "eliminar_rayos_cosmicos": cu.eliminar_rayos_cosmicos,
    "emsc": cu.emsc,
}


def aplicar_paso(X_in, paso_tup):
    if paso_tup is None:
        return X_in
    nombre, params = paso_tup
    if nombre in FUNCIONES_EXTRA:
        return FUNCIONES_EXTRA[nombre](X_in, **params)
    return cu.aplicar_pretratamientos(X_in, [(nombre, params)])


# =============================================================================
# SIDEBAR: data and class loading (always visible)
# =============================================================================

with st.sidebar:
    _logo_b64 = _img_b64("logo_espectrometrika_solo.png")
    st.markdown(f"""
    <div style="margin-bottom:2px;">
        <img src="{_logo_b64}" style="height:46px; display:block;">
    </div>
    """, unsafe_allow_html=True)
    st.caption("Preprocessing, exploratory analysis, classification & regression")

    _inti_b64 = _img_b64("inti_logo.png")
    if _inti_b64:
        st.markdown(f"""
        <div style="margin:10px 0 14px 0;">
            <img src="{_inti_b64}" style="height:46px;">
        </div>
        """, unsafe_allow_html=True)

    with st.expander("📖 Theoretical guide (PDF)"):
        st.caption("A companion reference covering every tool in this app — preprocessing per "
                   "instrument, PCA, outlier detection, SIMCA, variable selection, all the "
                   "classification/regression algorithms, MCR-ALS, validation strategies, every "
                   "metric with its formula, and how to read each diagnostic plot (Williams plot, "
                   "learning curves, ROC curve, statistical model comparison...). Useful to read "
                   "alongside your analysis to help interpret what you're seeing.")
        _ruta_guia = os.path.join(os.path.dirname(__file__), "assets", "theoretical_guide.pdf")
        try:
            with open(_ruta_guia, "rb") as _f:
                st.download_button(
                    "⬇️ Download theoretical guide (PDF)", data=_f.read(),
                    file_name="Introduction_to_Chemometrics.pdf", mime="application/pdf",
                )
        except FileNotFoundError:
            st.caption("(Guide file not found — make sure assets/theoretical_guide.pdf is included.)")

    with st.expander("📤 Load a saved project"):
        st.caption("Restores data, preprocessing, and saved models from a project file downloaded "
                   "earlier — the fastest way to pick up exactly where you left off. This is "
                   "available whether or not you already have data loaded (it overwrites whatever "
                   "is currently loaded).")
        archivo_proyecto = st.file_uploader(
            "Project file (.joblib)", type=["joblib"], key="cargar_proyecto",
        )
        if archivo_proyecto is not None and st.button("📤 Load this project"):
            try:
                proyecto_cargado = joblib.load(archivo_proyecto)
                claves_esperadas_proyecto = {
                    "df", "ids", "numeros_onda", "X", "clases", "mascara_excluidas",
                    "X_pret", "numeros_onda_pret", "pasos_pretratamiento",
                    "valores_y", "modelos_guardados",
                }
                claves_faltantes = claves_esperadas_proyecto - set(proyecto_cargado.keys())
                if len(claves_faltantes) > 2:
                    st.error("This file doesn't look like a project saved by this app.")
                else:
                    for k, v in proyecto_cargado.items():
                        st.session_state[k] = v
                    st.success("Project loaded successfully.")
                    st.rerun()
            except Exception as e:
                st.error(f"Could not load the project file: {e}")

    with st.expander("💬 Get in touch"):
        st.caption("Tell us about your experience with the app, or let us know about any "
                   "problems or improvements you'd like to see — we'd love to hear from you.")
        _mensaje_contacto = st.text_area(
            "Your message", key="contacto_mensaje", label_visibility="collapsed",
            placeholder="Write your message here...", height=100,
        )
        _col_nombre, _col_email = st.columns(2)
        with _col_nombre:
            _nombre_contacto = st.text_input("Your name (optional)", key="contacto_nombre")
        with _col_email:
            _email_contacto = st.text_input(
                "Your email (optional)", key="contacto_email",
                help="Only if you'd like a reply.",
            )
        if st.button("📨 Send message", use_container_width=True):
            if not _mensaje_contacto.strip():
                st.warning("Please write a message before sending.")
            else:
                _ok, _detalle = enviar_mensaje_contacto(
                    _mensaje_contacto, _nombre_contacto, _email_contacto,
                )
                if _ok:
                    st.success(_detalle)
                else:
                    st.error(_detalle)

    st.header("1. Load data")
    archivo = st.file_uploader(
        "Spectra file",
        type=["csv", "xlsx", "xls"],
        help="You can choose afterwards which row and which columns correspond to what.",
    )

    clases_desde_archivo = None
    valores_y_desde_archivo = None

    if archivo is not None:
        try:
            id_archivo = getattr(archivo, "file_id", None) or f"{archivo.name}_{archivo.size}"
            es_excel = not archivo.name.lower().endswith(".csv")
            bytes_archivo = archivo.getvalue()

            hoja_elegida = None
            if es_excel:
                nombres_hojas = pd.ExcelFile(io.BytesIO(bytes_archivo)).sheet_names
                if len(nombres_hojas) > 1:
                    hoja_elegida = st.selectbox(
                        "Sheet", nombres_hojas, key=f"hoja_{id_archivo}",
                        help="This Excel file has more than one sheet — pick which one to load.",
                    )
                else:
                    hoja_elegida = nombres_hojas[0]

            with st.expander("👁️ Raw preview (to choose the header row)"):
                df_crudo = _leer_preview_cacheada(bytes_archivo, archivo.name, hoja_elegida)
                st.dataframe(df_crudo, width='stretch')

            fila_encabezado = st.number_input(
                "Which row contains the wavenumbers / variable names?",
                min_value=1, max_value=10, value=1, step=1,
                help="1 = first row of the file. Increase it if your file has metadata above the real header.",
                key=f"fila_encabezado_{id_archivo}",
            )

            df_completo = _leer_archivo_cacheado(bytes_archivo, archivo.name, hoja_elegida, fila_encabezado)
            # Normalize column labels to text right away. Excel keeps numeric
            # header cells (e.g. wavenumbers) as actual int/float column labels,
            # while CSV headers are always text — without this, selecting a
            # column by its displayed name later would fail for Excel files
            # with numeric headers (KeyError: label not found).
            df_completo.columns = [str(c) for c in df_completo.columns]
            columnas = list(df_completo.columns)

            col1, col2 = st.columns(2)
            with col1:
                opcion_id = st.selectbox(
                    "Sample ID column",
                    ["(none — auto-generate)"] + columnas,
                    key=f"opcion_id_{id_archivo}",
                    help="Which column holds each sample's unique name/ID. Pick 'none' to auto-generate "
                         "Sample_1, Sample_2... if your file doesn't have one.",
                )
            with col2:
                opcion_clase = st.selectbox(
                    "Class column (for classification / SIMCA, if already in the file)",
                    ["(none)"] + columnas,
                    key=f"opcion_clase_{id_archivo}",
                    help="Which column holds the class/group label for each sample. Only use this for "
                         "a CATEGORICAL label (e.g. origin, variety). Leave as 'none' if you don't need "
                         "classification, or if your target is a continuous number — use the reference "
                         "value column below for that instead.",
                )
            opcion_valor_y = st.selectbox(
                "Reference value column (for regression, if already in the file)",
                ["(none)"] + columnas,
                key=f"opcion_valory_{id_archivo}",
                help="Which column holds the continuous numeric value you want to predict with "
                     "regression (e.g. a lab-measured concentration). This is different from the class "
                     "column above — use this one for numbers, not categories.",
            )

            columnas_restantes = columnas.copy()

            if opcion_id == "(none — auto-generate)":
                ids = np.array([f"Sample_{i+1}" for i in range(len(df_completo))])
            else:
                ids = df_completo[opcion_id].astype(str).to_numpy()
                columnas_restantes.remove(opcion_id)
                n_duplicados = len(ids) - len(set(ids))
                if n_duplicados > 0:
                    st.warning(f"⚠️ The ID column has {n_duplicados} repeated value(s) — sample IDs "
                               "should normally be unique (otherwise things like exported predictions "
                               "become hard to trace back to a specific sample). Making them unique by "
                               "appending a running number.")
                    contador = {}
                    ids_unicos = []
                    for i in ids:
                        contador[i] = contador.get(i, 0) + 1
                        ids_unicos.append(i if contador[i] == 1 else f"{i}_{contador[i]}")
                    ids = np.array(ids_unicos)

            if opcion_clase != "(none)":
                clases_desde_archivo = df_completo[opcion_clase].astype(str).to_numpy()
                if opcion_clase in columnas_restantes:
                    columnas_restantes.remove(opcion_clase)

            if opcion_valor_y != "(none)":
                try:
                    valores_y_desde_archivo = pd.to_numeric(df_completo[opcion_valor_y]).to_numpy(dtype=float)
                except (ValueError, TypeError):
                    st.error(f"Column '{opcion_valor_y}' has non-numeric values — it can't be used as "
                             "a regression reference value.")
                    valores_y_desde_archivo = None
                if opcion_valor_y in columnas_restantes:
                    columnas_restantes.remove(opcion_valor_y)

            numeros_onda = ejes_a_float(columnas_restantes)
            X = df_completo[columnas_restantes].to_numpy(dtype=float)

            cambio_tamano = (st.session_state.X is None) or (st.session_state.X.shape[0] != X.shape[0])
            archivo_realmente_nuevo = (
                st.session_state.get("archivo_cargado_id") is not None
                and st.session_state.get("archivo_cargado_id") != id_archivo
            )
            df_final = pd.DataFrame(X, index=ids, columns=numeros_onda)
            df_final.index.name = "id"

            st.session_state.df = df_final
            st.session_state.ids = ids
            st.session_state.numeros_onda = numeros_onda
            st.session_state.X = X
            if cambio_tamano:
                st.session_state.mascara_excluidas = np.zeros(X.shape[0], dtype=bool)
            if archivo_realmente_nuevo or cambio_tamano:
                # A genuinely different file was loaded (or the sample count changed) — clear
                # classes/reference values from the PREVIOUS dataset unless the new file itself
                # already supplies them, so stale labels from an unrelated dataset never linger.
                if clases_desde_archivo is None:
                    st.session_state.clases = None
                if valores_y_desde_archivo is None:
                    st.session_state.valores_y = None
            if clases_desde_archivo is not None:
                st.session_state.clases = clases_desde_archivo
            if valores_y_desde_archivo is not None:
                st.session_state.valores_y = valores_y_desde_archivo
            if archivo_realmente_nuevo:
                # A genuinely different file was loaded (not just a re-parse of the same one) —
                # clear any trained-model results from the PREVIOUS dataset so stale
                # tables/plots referencing old samples can't linger on screen. Saved models
                # (in "Predict" tab) are intentionally kept, since re-using an already-trained
                # model on a new file is a legitimate workflow.
                resetear_prefijo("clf_")
                resetear_prefijo("reg_")
                resetear_prefijo("simca_")
                resetear_prefijo("mcr_")
            st.session_state["archivo_cargado_id"] = id_archivo
            st.session_state["archivo_cargado_nombre"] = archivo.name

            st.success(f"{X.shape[0]} samples × {X.shape[1]} variables"
                       + (" · reference values loaded" if valores_y_desde_archivo is not None else ""))
            if archivo_realmente_nuevo:
                st.caption("ℹ️ New file detected — previous Classification/Regression/SIMCA results "
                           "were cleared. Saved models (for the Predict tab) were kept.")
        except Exception as e:
            st.error(f"Could not read '{archivo.name}' with that configuration: {e}")
            if st.session_state.get("archivo_cargado_nombre") not in (None, archivo.name):
                st.warning(f"⚠️ Still showing the previous file "
                           f"('{st.session_state['archivo_cargado_nombre']}') because this one could not be read. "
                           "Check the header row and the chosen columns above.")

    if hay_datos():
        st.header("2. Classes / groups (optional)")

        if clases_desde_archivo is not None:
            st.success(f"Classes loaded from the file: {', '.join(map(str, np.unique(clases_desde_archivo)))}")
            usar_otra_fuente = st.checkbox(
                "Define classes another way instead",
                help="Override the class column from the file and use one of the methods below "
                     "instead (extract from ID, type manually, or upload a separate file).",
            )
        else:
            usar_otra_fuente = True

        if usar_otra_fuente:
            modo_clases = st.radio(
                "How do you want to define the classes?",
                ["None", "Extract from ID (separator)", "Extract from ID (character position)",
                 "Enter manually", "Upload file (id, class)"],
                index=0 if clases_desde_archivo is None else 0,
                help="How to assign each sample to a class/group. Extracting from the ID works "
                     "when your sample names already encode the class (e.g. 'A_01', or 'AB01' by "
                     "character position). Otherwise type them in or upload a lookup file.",
            )

            if modo_clases == "None":
                pass  # no-op: keep whatever classes are already loaded (from a project, a
                      # previous file, etc.) — classes are only cleared when a genuinely new
                      # file is loaded (see 'archivo_realmente_nuevo' below), never just
                      # because this radio's default option happens to be showing.

            elif modo_clases == "Extract from ID (separator)":
                sep = st.text_input("Separator in the ID", value="_")
                posicion = st.number_input("Fragment position (0 = first)", min_value=0, value=0, step=1)
                try:
                    clases = np.array([str(i).split(sep)[posicion] for i in st.session_state.ids])
                    st.session_state.clases = clases
                    st.caption(f"Classes detected: {', '.join(map(str, np.unique(clases)))}")
                except Exception:
                    st.warning("Could not extract the class with that separator/position for every ID.")

            elif modo_clases == "Extract from ID (character position)":
                st.caption("For example, in the ID \"AB01\", positions 1 and 2 (\"AB\") could be the class.")
                c1, c2 = st.columns(2)
                pos_inicio = c1.number_input("Start position (1 = first character)", min_value=1, value=1, step=1)
                pos_fin = c2.number_input("End position", min_value=1, value=2, step=1)
                try:
                    clases = np.array([str(i)[pos_inicio - 1: pos_fin] for i in st.session_state.ids])
                    st.session_state.clases = clases
                    st.caption(f"Classes detected: {', '.join(map(str, np.unique(clases)))}")
                except Exception:
                    st.warning("Could not extract the class with those positions for every ID.")

            elif modo_clases == "Enter manually":
                st.caption("One class per sample, comma-separated, in the same order as the IDs:")
                st.code(", ".join(st.session_state.ids[:8]) + (", ..." if len(st.session_state.ids) > 8 else ""))
                texto = st.text_area("Classes (comma-separated)")
                if texto.strip():
                    lista = [c.strip() for c in texto.split(",")]
                    if len(lista) != len(st.session_state.ids):
                        st.error(f"You entered {len(lista)} classes but there are {len(st.session_state.ids)} samples.")
                    else:
                        st.session_state.clases = np.array(lista)

            elif modo_clases == "Upload file (id, class)":
                archivo_clases = st.file_uploader("CSV with columns: id, class", type=["csv"], key="clases_csv")
                if archivo_clases is not None:
                    try:
                        df_clases = pd.read_csv(archivo_clases, dtype=str)
                        df_clases = df_clases.set_index(df_clases.columns[0])
                        mapa = df_clases.iloc[:, 0].to_dict()
                        clases = np.array([mapa.get(i, "no_class") for i in st.session_state.ids])
                        st.session_state.clases = clases
                        if "no_class" in clases:
                            st.warning("Some IDs were not found in the class file (marked 'no_class').")
                    except Exception as e:
                        st.error(f"Error reading the class file: {e}")

        n_excluidas = int(st.session_state.mascara_excluidas.sum()) if st.session_state.mascara_excluidas is not None else 0
        if n_excluidas > 0:
            st.info(f"🚫 {n_excluidas} sample(s) excluded as outliers ('Outliers' tab).")

        with st.expander("💾 Save full project"):
            st.caption("Bundles your data, preprocessing settings, and every saved model into a "
                       "single file, so you can close the app and pick up exactly where you left "
                       "off — instead of re-uploading the spectra and retraining everything.")
            CLAVES_PROYECTO = [
                "df", "ids", "numeros_onda", "X", "clases", "mascara_excluidas",
                "X_pret", "numeros_onda_pret", "pasos_pretratamiento",
                "valores_y", "modelos_guardados",
            ]
            if st.button("💾 Prepare project file", help="Packages everything listed above into one downloadable file."):
                proyecto = {k: st.session_state[k] for k in CLAVES_PROYECTO if k in st.session_state}
                buffer_proyecto = io.BytesIO()
                joblib.dump(proyecto, buffer_proyecto)
                st.session_state["_proyecto_bytes"] = buffer_proyecto.getvalue()
            if "_proyecto_bytes" in st.session_state:
                st.download_button(
                    "⬇️ Download project file", data=st.session_state["_proyecto_bytes"],
                    file_name="chemometrics_project.joblib", mime="application/octet-stream",
                )


# =============================================================================
# MAIN BODY: tabs
# =============================================================================

_logo_top_b64 = _img_b64("logo_espectrometrika_solo.png")
st.markdown(f"""
<div style="margin-bottom:10px; margin-top:4px;">
    <img src="{_logo_top_b64}" style="height:30px; opacity:0.9; display:block;">
</div>
""", unsafe_allow_html=True)

tabs = st.tabs([
    "🏠 Home",
    "📈 Data",
    "🧪 Preprocessing",
    "🧭 PCA",
    "🚩 Outliers",
    "🌳 Dendrogram",
    "🛠️ Other tools",
    "🏷️ Classification",
    "🧬 SIMCA",
    "📉 Regression",
    "🔮 Prediction",
])

# -----------------------------------------------------------------------
# TAB: HOME
# -----------------------------------------------------------------------
with tabs[0]:
    _logo_hero_b64 = _img_b64("logo_espectrometrika_solo.png")
    st.markdown(f"""
    <div style="margin-bottom:6px;">
        <img src="{_logo_hero_b64}" style="height:95px; display:block;">
    </div>
    <p style="font-family:'Inter',sans-serif; font-size:1.05rem; color:#4A5A63; max-width:820px;
              margin-top:2px; margin-bottom:28px; line-height:1.5;">
        A free, open initiative bringing chemometric analysis to students, professionals, and
        analytical chemistry practitioners.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("""
    A complete chemometrics platform for spectroscopy and chromatography data —
    from raw signal to a trained, validated, and traceable predictive model.
    """)

    def _tarjeta(titulo, texto, bg, borde):
        st.markdown(f"""
        <div style="background:{bg}; border-left:4px solid {borde}; border-radius:10px;
                    padding:18px 20px; min-height:210px;">
            <div style="font-family:'Inter',sans-serif; font-weight:700; font-size:1.02rem;
                        color:#0B3D54; margin-bottom:8px;">{titulo}</div>
            <div style="font-family:'Inter',sans-serif; font-size:0.92rem; color:#3D4A52;
                        line-height:1.5;">{texto}</div>
        </div>
        """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        _tarjeta(
            "🔍 Exploratory analysis",
            "Preprocessing for NIR, MIR, Raman, chromatograms and NMR, PCA, outlier "
            "detection (Hotelling's T² / Q), hierarchical clustering and interactive "
            "visualizations.",
            TEAL_SOFT, TEAL,
        )
    with col2:
        _tarjeta(
            "🏷️ Classification & 📉 Regression",
            "10+ algorithms per task (LDA, PLS-DA, Random Forest, SVM, XGBoost, neural "
            "networks, PLS, Ridge/Lasso, and more), variable selection (Boruta / genetic "
            "algorithm) and hyperparameter optimization.",
            "#EAF2F7", PETROLEUM_LIGHT,
        )
    with col3:
        _tarjeta(
            "🔮 Prediction & traceability",
            "Apply a trained model to brand-new samples, with automatic axis "
            "interpolation and a downloadable model card (data, settings, metrics, "
            "software versions) for every model you save.",
            "#FDF6E9", "#D97706",
        )

    if hay_datos():
        st.markdown("""
        <div style="margin-top:26px;">
        ✅ You already have a dataset loaded — use the tabs above to continue working with it.
        </div>
        """, unsafe_allow_html=True)
        if st.button("🗑️ Clear data and start over", key="btn_reset_home"):
            st.session_state.clear()
            st.rerun()
    else:
        st.markdown("""
        <div style="margin-top:26px;">
        👈 To get started, upload your spectra file from the left-hand panel — the
        app will help you identify the header row and the ID/class columns.
        </div>
        """, unsafe_allow_html=True)

    _hero_img_b64 = _img_b64("hero_spectra.png")
    if _hero_img_b64:
        st.markdown(f"""
        <div style="margin-top:36px; text-align:center;">
            <img src="{_hero_img_b64}" style="max-width:78%; opacity:0.92;">
        </div>
        """, unsafe_allow_html=True)

if not hay_datos():
    # All the other tabs need a loaded dataset to make sense — show a friendly
    # placeholder in each of them (so they're not just blank if clicked) and
    # stop here, before their real logic below (which assumes data exists).
    for _tab_vacia in tabs[1:]:
        with _tab_vacia:
            st.info("📁 Upload a spectra file in the sidebar to get started.")
    st.stop()

# -----------------------------------------------------------------------
# TAB: DATA
# -----------------------------------------------------------------------
with tabs[1]:
    st.subheader("Data preview")
    st.dataframe(st.session_state.df.head(10), width='stretch')

    ids, X, clases = datos_activos()

    st.subheader("Spectra")
    fig = go.Figure()
    colores_clase = None
    if clases is not None:
        paleta = CLASS_PALETTE
        clases_unicas = np.unique(clases)
        colores_clase = {c: paleta[i % len(paleta)] for i, c in enumerate(clases_unicas)}

    for i in range(X.shape[0]):
        color = colores_clase[clases[i]] if colores_clase is not None else "steelblue"
        fig.add_trace(go.Scatter(
            x=st.session_state.numeros_onda, y=X[i], mode="lines",
            line=dict(width=1, color=color), opacity=0.6,
            name=str(clases[i]) if clases is not None else str(ids[i]),
            legendgroup=str(clases[i]) if clases is not None else None,
            showlegend=bool(clases is not None and clases[i] not in [t.name for t in fig.data]),
            hovertext=str(ids[i]),
        ))
    fig.update_layout(height=450, xaxis_title="Wavenumber / wavelength", yaxis_title="Signal")
    if st.session_state.numeros_onda[0] > st.session_state.numeros_onda[-1]:
        fig.update_xaxes(autorange="reversed")
    st.plotly_chart(fig, width='stretch')
    st.caption(f"Showing {X.shape[0]} of {st.session_state.X.shape[0]} samples "
               f"({int(st.session_state.mascara_excluidas.sum())} excluded as outliers).")

# -----------------------------------------------------------------------
# TAB: PREPROCESSING
# -----------------------------------------------------------------------
with tabs[2]:
    st.subheader("Spectral preprocessing")

    # Important: preprocessing is always computed on ALL samples (not just the
    # active ones), because further down (PCA, Outliers, Dendrogram, etc.) this
    # same result is filtered with the exclusion mask. If it came pre-filtered
    # here, that second filtering would break as soon as an outlier is excluded
    # (mismatched sizes).
    ids, X, clases = st.session_state.ids, st.session_state.X, st.session_state.clases
    numeros_onda_crudo = st.session_state.numeros_onda

    tipo_senal = st.radio(
        "Signal type",
        ["NIR / MIR", "Raman", "Chromatogram", "NMR"],
        horizontal=True,
        help="Changes which preprocessing options are available below.",
    )

    # --- Step 0: NMR only (bucketing). Also changes the spectral axis. ---
    X_paso0 = X
    numeros_onda_paso0 = numeros_onda_crudo
    if tipo_senal == "NMR":
        usar_bucketing = st.checkbox(
            "Apply bucketing (binning)", value=True,
            help="Groups the high-resolution NMR spectrum into fixed-width bins, summing the signal "
                 "within each — drastically reduces the number of variables and makes the data more "
                 "robust to small peak shifts between samples.",
        )
        if usar_bucketing:
            rango_eje = abs(float(numeros_onda_crudo[-1]) - float(numeros_onda_crudo[0]))
            ancho_default = max(rango_eje / 300, 1e-6)
            ancho_min = max(rango_eje / 3000, 1e-6)
            ancho_max = max(rango_eje / 10, ancho_default * 2)
            ancho_bucket = st.slider(
                "Bucket width (same units as your spectral axis, e.g. ppm or Hz)",
                min_value=float(ancho_min), max_value=float(ancho_max),
                value=float(ancho_default), format="%.5f",
                help="Wider buckets = fewer variables and more tolerance to peak shifts, but less "
                     "resolution (nearby peaks can merge). Narrower buckets keep more detail but "
                     "are more sensitive to small shifts between samples.",
            )
            try:
                X_paso0, numeros_onda_paso0 = cu.bucketing(X, numeros_onda_crudo, ancho_bucket)
                st.caption(f"Bucketing applied: {X.shape[1]} variables → {X_paso0.shape[1]} buckets.")
                if X_paso0.shape[1] > 3000:
                    st.warning(
                        f"Bucketing left {X_paso0.shape[1]} variables — that's a lot for "
                        "a small dataset. Try increasing the bucket width."
                    )
            except Exception as e:
                st.error(f"Bucketing failed: {e}")

    st.caption("Build a combination in 2 steps: first a scatter/scale/baseline "
               "correction, then a Savitzky-Golay filter (when it applies to the "
               "chosen signal type).")

    opciones_a_por_tipo = {
        "NIR / MIR": ["None", "Normalization (area)", "Normalization (vector)",
                      "Standardization", "SNV", "MSC", "EMSC"],
        "Raman": ["None", "Remove cosmic rays", "Baseline correction (ALS)",
                  "SNV", "Standardization"],
        "Chromatogram": ["None", "Baseline correction (ALS)",
                         "Normalization (area)", "Standardization"],
        "NMR": ["None", "Normalization (PQN)", "Standardization", "SNV"],
    }
    opciones_b_por_tipo = {
        "NIR / MIR": ["None", "Smoothing (order 0)", "1st derivative", "2nd derivative"],
        "Raman": ["None", "Smoothing (order 0)", "1st derivative", "2nd derivative"],
        "Chromatogram": ["None"],
        "NMR": ["None"],
    }

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        paso_a = st.selectbox("Step A", opciones_a_por_tipo[tipo_senal])
    with col2:
        paso_b = st.selectbox("Step B — Savitzky-Golay", opciones_b_por_tipo[tipo_senal])
    with col3:
        orden = st.radio("Application order", ["A → B (recommended)", "B → A"])

    if paso_a == "Baseline correction (ALS)":
        c1, c2 = st.columns(2)
        lam_als = c1.select_slider("Baseline smoothness (lambda)",
                                    options=[1e3, 1e4, 1e5, 1e6, 1e7, 1e8], value=1e5)
        p_als = c2.slider("Asymmetry (p)", min_value=0.001, max_value=0.1, value=0.01, step=0.001)
    if paso_a == "Remove cosmic rays":
        c1, c2 = st.columns(2)
        ventana_rc = c1.slider("Median filter window (odd)", min_value=3, max_value=15, value=5, step=2)
        umbral_rc = c2.slider("Threshold (modified z-score)", min_value=3, max_value=15, value=7)
    if paso_a == "EMSC":
        orden_emsc = st.slider(
            "Polynomial order (wavelength-dependent effects)", min_value=0, max_value=4, value=2,
            help="How many polynomial terms of the wavelength axis to model and remove, on top of "
                 "the usual offset + scaling that plain MSC already corrects for. Order 0 behaves "
                 "like plain MSC; higher orders can correct sloped or curved baseline effects that "
                 "vary smoothly across the spectrum, at the cost of possibly removing some real "
                 "chemical signal if set too high.",
        )

    if paso_b != "None":
        c1, c2 = st.columns(2)
        ventana = c1.slider("SG window", min_value=5, max_value=51, value=11, step=2)
        orden_poly = c2.slider("Polynomial order", min_value=1, max_value=5, value=2)
    else:
        ventana, orden_poly = 11, 2

    mapa_a = {
        "Normalization (area)": ("normalizar", {"modo": "area"}),
        "Normalization (vector)": ("normalizar", {"modo": "vector"}),
        "Normalization (PQN)": ("pqn", {}),
        "Standardization": ("estandarizar", {}),
        "SNV": ("snv", {}),
        "MSC": ("msc", {}),
        "EMSC": ("emsc", {"eje": numeros_onda_paso0, "orden_polinomio": orden_emsc} if paso_a == "EMSC" else {}),
        "Baseline correction (ALS)": (
            "linea_base_als", {"lam": lam_als, "p": p_als} if paso_a == "Baseline correction (ALS)" else {}
        ),
        "Remove cosmic rays": (
            "eliminar_rayos_cosmicos", {"ventana": ventana_rc, "umbral": umbral_rc}
            if paso_a == "Remove cosmic rays" else {}
        ),
    }
    mapa_b = {
        "Smoothing (order 0)": ("suavizado_sg", {"ventana": ventana, "orden_polinomio": orden_poly}),
        "1st derivative": ("derivada_sg", {"orden": 1, "ventana": ventana, "orden_polinomio": orden_poly}),
        "2nd derivative": ("derivada_sg", {"orden": 2, "ventana": ventana, "orden_polinomio": orden_poly}),
    }

    # (aplicar_paso and FUNCIONES_EXTRA are defined above, at module level)

    paso_a_tup = mapa_a.get(paso_a)
    paso_b_tup = mapa_b.get(paso_b)
    secuencia = [paso_a_tup, paso_b_tup] if orden.startswith("A") else [paso_b_tup, paso_a_tup]

    error_pretratamiento = None
    try:
        X_pret = _aplicar_pretratamiento_cacheado(X_paso0, tuple(secuencia))
    except Exception as e:
        error_pretratamiento = str(e)
        X_pret = X_paso0.copy()

    if error_pretratamiento:
        st.error(
            "Could not apply the preprocessing (check the chosen parameters, "
            "e.g. that the SG window is odd and larger than the polynomial order). "
            f"Details: {error_pretratamiento}"
        )

    st.session_state.X_pret = X_pret
    st.session_state.numeros_onda_pret = numeros_onda_paso0  # effective axis for the rest of the app
    st.session_state.pasos_pretratamiento = [p for p in secuencia if p is not None]

    st.markdown("**Before / after comparison**")
    col_izq, col_der = st.columns(2)
    with col_izq:
        fig1 = go.Figure()
        for i in range(min(X.shape[0], 60)):
            fig1.add_trace(go.Scatter(x=numeros_onda_crudo, y=X[i],
                                       mode="lines", line=dict(width=0.8), opacity=0.5,
                                       showlegend=False))
        fig1.update_layout(title="Before", height=380)
        if numeros_onda_crudo[0] > numeros_onda_crudo[-1]:
            fig1.update_xaxes(autorange="reversed")
        st.plotly_chart(fig1, width='stretch')
    with col_der:
        fig2 = go.Figure()
        for i in range(min(X_pret.shape[0], 60)):
            fig2.add_trace(go.Scatter(x=numeros_onda_paso0, y=X_pret[i],
                                       mode="lines", line=dict(width=0.8), opacity=0.5,
                                       showlegend=False))
        fig2.update_layout(title="After", height=380)
        if numeros_onda_paso0[0] > numeros_onda_paso0[-1]:
            fig2.update_xaxes(autorange="reversed")
        st.plotly_chart(fig2, width='stretch')

    nombres_pasos_es = [p[0] for p in secuencia if p is not None]
    nombres_mostrar = []
    if orden.startswith("A"):
        if paso_a != "None": nombres_mostrar.append(paso_a)
        if paso_b != "None": nombres_mostrar.append(paso_b)
    else:
        if paso_b != "None": nombres_mostrar.append(paso_b)
        if paso_a != "None": nombres_mostrar.append(paso_a)
    if nombres_mostrar:
        st.caption("Combination applied: " + " → ".join(nombres_mostrar))
    else:
        st.caption("No preprocessing (using raw spectra).")

    st.info("This preprocessing is automatically used across the rest of the tabs "
            "(PCA, Outliers, Dendrogram, Classification, Regression, etc.).")

    df_exportar = cu.armar_dataframe_exportable(ids, numeros_onda_paso0, X_pret)
    st.dataframe(df_exportar, width='stretch', height=220)
    col_dl1, col_dl2 = st.columns(2)
    col_dl1.download_button(
        "⬇️ Download preprocessed dataset (CSV)",
        data=df_exportar.to_csv().encode("utf-8"),
        file_name="preprocessed_spectra.csv",
        mime="text/csv",
    )
    col_dl2.download_button(
        "⬇️ Download preprocessed dataset (Excel)",
        data=df_a_excel_bytes({"preprocessed_spectra": df_exportar.reset_index(names="id")}),
        file_name="preprocessed_spectra.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="descargar_pretratado_excel",
    )

# -----------------------------------------------------------------------
# TAB: PCA
# -----------------------------------------------------------------------
with tabs[3]:
    st.subheader("Principal Component Analysis (PCA)")

    ids_actuales, _, clases_actuales = datos_activos()
    X_pca_input = st.session_state.X_pret[indice_activo()]
    n_muestras, n_variables = X_pca_input.shape
    n_comp_max = min(n_muestras - 1, n_variables)

    if n_comp_max < 2:
        st.warning("At least 3 samples are needed to compute PCA.")
    else:
        esta_desactualizado_pca = insignia_estado("pca_firma", "The PCA result")
        col_btn_pca, _ = st.columns([1, 3])
        with col_btn_pca:
            _texto_boton_pca = "▶ Compute PCA" if st.session_state.get("pca_completo") is None else "🔄 Update PCA"
            if st.button(_texto_boton_pca, type="primary" if esta_desactualizado_pca else "secondary",
                         key="btn_computar_pca"):
                with st.spinner("Fitting PCA..."):
                    _pca_completo = _ajustar_pca_cacheado(X_pca_input, n_comp_max)
                    st.session_state.pca_completo = _pca_completo
                    st.session_state.scores_completo = _pca_completo.transform(X_pca_input)
                    st.session_state.cargas_completo = _pca_completo.components_
                    st.session_state.autovalores = _pca_completo.explained_variance_
                    st.session_state.var_explicada = _pca_completo.explained_variance_ratio_ * 100
                    # IMPORTANT: freeze EVERYTHING this PCA result is consistent with —
                    # if outlier exclusions change afterwards (without clicking Update),
                    # using live data/ids anywhere downstream would silently mismatch
                    # this frozen result in length. Outliers (next tab) reuses all of this.
                    st.session_state.pca_X_input = X_pca_input
                    st.session_state.pca_ids = ids_actuales
                    st.session_state.pca_clases = clases_actuales
                    st.session_state.pca_firma = firma_datos_activos()
                st.rerun()

        if st.session_state.get("pca_completo") is None:
            st.info("Click **Compute PCA** above to get started.")
        else:
            pca_completo = st.session_state.pca_completo
            scores_completo = st.session_state.scores_completo
            cargas_completo = st.session_state.cargas_completo
            autovalores = st.session_state.autovalores
            ids = st.session_state.pca_ids
            clases = st.session_state.pca_clases
            var_explicada = st.session_state.var_explicada
            var_acumulada = np.cumsum(var_explicada)
            # Use the STORED result's own dimensions from here on, not the live
            # dataset's — if outliers were excluded/restored after this PCA was
            # computed (without clicking Update), the live data could have a
            # different number of samples/components than this stale result.
            n_comp_max_guardado = len(var_explicada)

            col1, col2 = st.columns([2, 1])
            with col1:
                fig = go.Figure()
                fig.add_bar(x=list(range(1, len(var_explicada) + 1)), y=var_explicada, name="% individual")
                fig.add_trace(go.Scatter(x=list(range(1, len(var_acumulada) + 1)), y=var_acumulada,
                                          mode="lines+markers", name="% cumulative", yaxis="y2"))
                fig.update_layout(
                    title="Explained variance (scree plot)",
                    xaxis_title="Principal component",
                    yaxis=dict(title="% individual"),
                    yaxis2=dict(title="% cumulative", overlaying="y", side="right"),
                    xaxis=dict(range=[0.5, min(15, len(var_explicada)) + 0.5]),
                    height=380,
                )
                st.plotly_chart(fig, width='stretch')
            with col2:
                n_comp = st.slider("Components to retain (n_comp)", min_value=2,
                                    max_value=min(10, n_comp_max_guardado), value=min(3, n_comp_max_guardado),
                                    help="How many principal components to keep for scores/loadings plots "
                                         "and for outlier detection (T²/Q). Check the scree plot on the "
                                         "left: pick enough to capture most of the variance, without "
                                         "including components that just look like noise.")
                st.metric("Cumulative explained variance", f"{var_acumulada[n_comp - 1]:.1f}%")
                st.session_state.n_comp = n_comp

            scores = scores_completo[:, :n_comp]
            cargas = cargas_completo[:n_comp, :]
            st.session_state.scores = scores
            st.session_state.cargas = cargas

            def grafico_scores(pc_x, pc_y):
                fig = px.scatter(
                    x=scores_completo[:, pc_x - 1], y=scores_completo[:, pc_y - 1],
                    color=clases if clases is not None else None,
                    color_discrete_sequence=CLASS_PALETTE,
                    hover_name=ids,
                    labels={"x": f"PC{pc_x} ({var_explicada[pc_x-1]:.1f}%)",
                            "y": f"PC{pc_y} ({var_explicada[pc_y-1]:.1f}%)"},
                    title=f"Scores: PC{pc_x} vs PC{pc_y}",
                )
                fig.add_hline(y=0, line_color="lightgray")
                fig.add_vline(x=0, line_color="lightgray")
                fig.update_layout(height=420)
                return fig

            n_comp_disponibles = min(15, n_comp_max_guardado)
            opciones_pc = list(range(1, n_comp_disponibles + 1))
            st.markdown("**2D scores plots** — pick any pair of components to compare "
                         f"(up to PC{n_comp_disponibles}).")
            c1, c2 = st.columns(2)
            with c1:
                cc1, cc2 = st.columns(2)
                pcx_1 = cc1.selectbox("X axis", opciones_pc, index=0, key="pcx_1",
                                       help="Which principal component to plot on the horizontal axis.")
                pcy_1 = cc2.selectbox("Y axis", opciones_pc, index=min(1, n_comp_disponibles - 1), key="pcy_1",
                                       help="Which principal component to plot on the vertical axis.")
                st.plotly_chart(grafico_scores(pcx_1, pcy_1), width='stretch')
            with c2:
                cc3, cc4 = st.columns(2)
                idx_default_x2 = min(1, n_comp_disponibles - 1)
                idx_default_y2 = min(2, n_comp_disponibles - 1)
                pcx_2 = cc3.selectbox("X axis", opciones_pc, index=idx_default_x2, key="pcx_2")
                pcy_2 = cc4.selectbox("Y axis", opciones_pc, index=idx_default_y2, key="pcy_2")
                st.plotly_chart(grafico_scores(pcx_2, pcy_2), width='stretch')

            st.markdown("**Interactive 3D plot (drag to rotate)**")
            if n_comp_max_guardado >= 3:
                cc5, cc6, cc7 = st.columns(3)
                pcx_3d = cc5.selectbox("X axis", opciones_pc, index=0, key="pcx_3d")
                pcy_3d = cc6.selectbox("Y axis", opciones_pc, index=min(1, n_comp_disponibles - 1), key="pcy_3d")
                pcz_3d = cc7.selectbox("Z axis", opciones_pc, index=min(2, n_comp_disponibles - 1), key="pcz_3d")
                col_x, col_y, col_z = f"PC{pcx_3d}", f"PC{pcy_3d}", f"PC{pcz_3d}"
                df3d = pd.DataFrame({
                    col_x: scores_completo[:, pcx_3d - 1],
                    col_y: scores_completo[:, pcy_3d - 1],
                    col_z: scores_completo[:, pcz_3d - 1],
                })
                df3d["ID"] = ids
                df3d["Class"] = clases if clases is not None else "All samples"
                fig3d = px.scatter_3d(df3d, x=col_x, y=col_y, z=col_z, color="Class", hover_name="ID",
                                       color_discrete_sequence=CLASS_PALETTE)
                fig3d.update_traces(marker=dict(size=5))
                fig3d.update_layout(height=550)
                st.plotly_chart(fig3d, width='stretch')
            else:
                st.caption("At least 3 components are needed for the 3D plot.")

# -----------------------------------------------------------------------
# TAB: OUTLIERS
# -----------------------------------------------------------------------
with tabs[4]:
    st.subheader("Outlier detection: Hotelling's T² and Q residual")

    if st.session_state.get("pca_completo") is None:
        st.info("Compute PCA first, in the **PCA** tab (needs at least 2 components).")
    else:
        # Use the SAME frozen snapshot the PCA result itself was computed from —
        # not live data — so they always stay consistent in length with each other,
        # regardless of outlier exclusions made afterwards without clicking Update.
        ids = st.session_state.pca_ids
        clases = st.session_state.pca_clases
        X_pca_input = st.session_state.pca_X_input
        scores_completo = st.session_state.scores_completo
        cargas_completo = st.session_state.cargas_completo
        autovalores = st.session_state.autovalores
        n_comp = min(st.session_state.n_comp, len(st.session_state.var_explicada))
        n_muestras = X_pca_input.shape[0]

        col1, col2, col3 = st.columns(3)
        alpha = col1.slider("Significance level (alpha)", 0.01, 0.10, 0.05, step=0.01)
        criterio = col2.selectbox("Criterion to flag outliers",
                                   ["T² and Q (both, conservative)", "T² or Q (either, aggressive)",
                                    "T² only", "Q only"])

        _firma_outliers_actual = firma_datos_activos() + (n_comp, alpha)
        _resultado_outliers_previo = st.session_state.get("outliers_resultado")
        esta_desactualizado_out = (
            _resultado_outliers_previo is None
            or st.session_state.get("outliers_firma") != _firma_outliers_actual
        )
        if _resultado_outliers_previo is not None:
            if esta_desactualizado_out:
                st.warning("⚠️ **This outlier result is out of date** — PCA, preprocessing, alpha, or "
                           "n_comp changed since it was last computed. The plot below still shows the "
                           "last computed result. Click **Update** to recompute it.", icon="⚠️")
            else:
                st.caption("✅ Up to date with the current PCA result, preprocessing, and settings.")
        _texto_boton_out = "▶ Compute outliers" if _resultado_outliers_previo is None else "🔄 Update outliers"
        if st.button(_texto_boton_out, type="primary" if esta_desactualizado_out else "secondary",
                     key="btn_computar_outliers"):
            with st.spinner("Computing T² / Q..."):
                st.session_state.outliers_resultado = _calcular_outliers_cacheado(
                    X_pca_input, scores_completo, cargas_completo, autovalores,
                    st.session_state.pca_completo.mean_, n_comp, alpha,
                )
                # Freeze the ids/classes THIS result goes with. PCA can be updated
                # independently afterwards (different sample count) — if that
                # happens before Outliers is also updated, using PCA's "current"
                # ids here instead of this frozen snapshot would mismatch T2/Q below.
                st.session_state.outliers_ids = ids
                st.session_state.outliers_clases = clases
                st.session_state.outliers_firma = _firma_outliers_actual
            st.rerun()

        if st.session_state.get("outliers_resultado") is None:
            st.info("Click **Compute outliers** above to get started.")
        else:
            ids = st.session_state.outliers_ids
            clases = st.session_state.outliers_clases
            T2, T2_lim, Q, Q_lim, Q_lim_confiable = st.session_state.outliers_resultado
            if not Q_lim_confiable:
                st.warning(
                    "⚠ The theoretical Q limit (Jackson-Mudholkar) is not reliable with this "
                    "'n_comp' (happens when almost all the variance is already explained). "
                    "An EMPIRICAL limit (95th percentile of your own Q values) is used instead."
                )

            col3.metric("T² limit", f"{T2_lim:.2f}")
            col3.metric("Q limit", f"{Q_lim:.4g}" + (" (empirical)" if not Q_lim_confiable else ""))

            T2_lo, T2_hi = cu.rango_con_margen(T2, T2_lim)
            Q_lo, Q_hi = cu.rango_con_margen(Q, Q_lim)

            fig = px.scatter(x=T2, y=Q, hover_name=ids,
                              color=clases if clases is not None else None,
                    color_discrete_sequence=CLASS_PALETTE,
                              labels={"x": "Hotelling's T²", "y": "Q residual"})
            fig.add_shape(type="rect", x0=T2_lo, x1=T2_lim, y0=Q_lo, y1=Q_lim,
                          fillcolor="green", opacity=0.08, line_width=0, layer="below")
            fig.add_shape(type="rect", x0=T2_lim, x1=T2_hi, y0=Q_lo, y1=Q_lim,
                          fillcolor="orange", opacity=0.08, line_width=0, layer="below")
            fig.add_shape(type="rect", x0=T2_lo, x1=T2_lim, y0=Q_lim, y1=Q_hi,
                          fillcolor="orange", opacity=0.08, line_width=0, layer="below")
            fig.add_shape(type="rect", x0=T2_lim, x1=T2_hi, y0=Q_lim, y1=Q_hi,
                          fillcolor="red", opacity=0.12, line_width=0, layer="below")
            fig.add_vline(x=T2_lim, line_dash="dash", line_color="red")
            fig.add_hline(y=Q_lim, line_dash="dash", line_color="red")
            fig.add_annotation(x=(T2_lo + T2_lim) / 2, y=(Q_lo + Q_lim) / 2, text="OK",
                                showarrow=False, font=dict(color="green"))
            fig.add_annotation(x=(T2_lim + T2_hi) / 2, y=(Q_lo + Q_lim) / 2, text="Outlier T²",
                                showarrow=False, font=dict(color="darkorange"))
            fig.add_annotation(x=(T2_lo + T2_lim) / 2, y=(Q_lim + Q_hi) / 2, text="Outlier Q",
                                showarrow=False, font=dict(color="darkorange"))
            fig.add_annotation(x=(T2_lim + T2_hi) / 2, y=(Q_lim + Q_hi) / 2, text="Outlier T² & Q",
                                showarrow=False, font=dict(color="darkred"))
            fig.update_xaxes(range=[T2_lo, T2_hi])
            fig.update_yaxes(range=[Q_lo, Q_hi])
            fig.update_layout(height=520, title="Influence plot")
            st.plotly_chart(fig, width='stretch')
            st.caption("Axes zoom automatically to your data (T² and Q are rarely exactly 0).")

            if criterio.startswith("T² and Q"):
                es_outlier = (T2 > T2_lim) & (Q > Q_lim)
            elif criterio.startswith("T² or Q"):
                es_outlier = (T2 > T2_lim) | (Q > Q_lim)
            elif criterio == "T² only":
                es_outlier = T2 > T2_lim
            else:
                es_outlier = Q > Q_lim

            candidatos = ids[es_outlier]
            st.markdown(f"**Outlier candidates with this criterion ({len(candidatos)}):** "
                        + (", ".join(candidatos) if len(candidatos) else "none"))

            if clases is not None and len(candidatos) > 0:
                conteo_candidatos_por_clase = pd.Series(clases[es_outlier]).value_counts()
                conteo_total_por_clase = pd.Series(clases).value_counts()
                proporciones = (conteo_candidatos_por_clase / conteo_total_por_clase).dropna()
                clases_con_muchos = proporciones[proporciones > 0.3].index.tolist()
                if clases_con_muchos:
                    st.info(f"ℹ️ A large share of class(es) **{', '.join(clases_con_muchos)}** show up here. "
                            "This outlier detection compares every sample to a SINGLE overall model — if a "
                            "whole class is chemically quite different from the rest, much of it can look "
                            "like an 'outlier' even though it's really just a distinct, valid population. "
                            "If that looks like what's happening, consider the **SIMCA** tab instead: it "
                            "builds a separate model per class, so a class won't be penalized just for "
                            "being different from the others.")

            col_a, col_b = st.columns(2)
            if col_a.button("🚫 Exclude these samples from the analysis", disabled=(len(candidatos) == 0)):
                idx_global = np.array([np.where(st.session_state.ids == i)[0][0] for i in candidatos])
                st.session_state.mascara_excluidas[idx_global] = True
                st.success(f"{len(candidatos)} sample(s) excluded. Other tabs no longer include them.")
                st.rerun()
            if col_b.button("♻️ Restore all samples (undo exclusions)"):
                st.session_state.mascara_excluidas[:] = False
                st.success("All samples restored.")
                st.rerun()

            st.divider()
            st.markdown("**📄 Exploratory analysis report**")
            st.caption("Includes: spectra, PCA (explained variance and scores), and the T²/Q "
                       "outlier plot with the critical limits and the candidate samples.")
            if st.button("🖨️ Generate report (PDF)", key="generar_reporte_exp"):
                var_explicada = st.session_state.var_explicada
                secciones_rep_exp = [
                    {"tipo": "titulo", "texto": "1. Dataset"},
                    {"tipo": "clave_valor", "pares": [
                        ("Samples used", int(X_pca_input.shape[0])),
                        ("Variables", int(X_pca_input.shape[1])),
                        ("Preprocessing", " -> ".join(p[0] for p in st.session_state.pasos_pretratamiento) or "none"),
                    ]},
                    {"tipo": "imagen", "fig": ru.fig_espectros(
                        st.session_state.numeros_onda_pret, X_pca_input,
                        clases=como_texto(clases) if clases is not None else None, titulo="Preprocessed spectra")},
                    {"tipo": "titulo", "texto": "2. PCA"},
                    {"tipo": "clave_valor", "pares": [
                        ("Components retained", n_comp),
                        ("Cumulative explained variance", f"{np.cumsum(var_explicada)[n_comp-1]:.1f}%"),
                    ]},
                    {"tipo": "imagen", "fig": ru.fig_scree(var_explicada, n_comp_marcado=n_comp)},
                    {"tipo": "imagen", "fig": ru.fig_scores(
                        scores_completo, var_explicada, 1, 2,
                        clases=como_texto(clases) if clases is not None else None, ids=ids)},
                    {"tipo": "salto_pagina"},
                    {"tipo": "titulo", "texto": "3. Outlier detection (Hotelling's T² and Q residual)"},
                    {"tipo": "clave_valor", "pares": [
                        ("Significance level (alpha)", alpha),
                        ("T² limit", f"{T2_lim:.2f}"),
                        ("Q limit", f"{Q_lim:.4g}" + (" (empirical)" if not Q_lim_confiable else " (theoretical)")),
                        ("Criterion used", criterio),
                        ("Outlier candidate samples", ", ".join(candidatos) if len(candidatos) else "none"),
                        ("Currently excluded samples", ", ".join(st.session_state.ids[st.session_state.mascara_excluidas])
                         if st.session_state.mascara_excluidas.any() else "none"),
                    ]},
                    {"tipo": "imagen", "fig": ru.fig_outliers(
                        T2, Q, T2_lim, Q_lim, T2_lo, T2_hi, Q_lo, Q_hi,
                        clases=como_texto(clases) if clases is not None else None)},
                    {"tipo": "salto_pagina"},
                    {"tipo": "titulo", "texto": "4. Hierarchical Cluster Analysis (HCA)"},
                    {"tipo": "clave_valor", "pares": [("Linkage method", "ward")]},
                ]
                if X_pca_input.shape[0] >= 3:
                    Z_reporte = hierarchy.linkage(X_pca_input, method="ward")
                    secciones_rep_exp.append({"tipo": "imagen", "fig": ru.fig_dendrograma(
                        Z_reporte, list(ids), clases=como_texto(clases) if clases is not None else None)})
                pdf_reporte_exp = ru.generar_reporte(
                    "Exploratory Analysis Report",
                    f"{X_pca_input.shape[0]} samples - {n_comp} principal components", secciones_rep_exp,
                )
                st.download_button(
                    "⬇️ Download full report (PDF)",
                    data=bytes(pdf_reporte_exp.output()),
                    file_name="exploratory_analysis_report.pdf", mime="application/pdf",
                    key="descargar_reporte_exp",
                )

# -----------------------------------------------------------------------
# TAB: DENDROGRAM
# -----------------------------------------------------------------------
with tabs[5]:
    st.subheader("Hierarchical Cluster Analysis (HCA)")

    ids, _, clases = datos_activos()
    X_hca = st.session_state.X_pret[indice_activo()]

    if X_hca.shape[0] < 3:
        st.warning("At least 3 samples are needed.")
    else:
        col1, col2 = st.columns(2)
        metodo = col1.selectbox(
            "Linkage method", ["ward", "average", "complete", "single"],
            help="How the distance between two CLUSTERS (not individual samples) is defined when "
                 "deciding what to merge next:\n"
                 "• ward: merges the pair that increases within-cluster variance the least — usually "
                 "gives the most balanced, compact clusters (good default).\n"
                 "• average: distance = average distance between all pairs across the two clusters.\n"
                 "• complete: distance = the FARTHEST pair between the two clusters — tends to give "
                 "tight, evenly-sized clusters, sensitive to outliers.\n"
                 "• single: distance = the CLOSEST pair between the two clusters — can chain together "
                 "long, straggly clusters (sensitive to noise, but can find elongated shapes).",
        )
        vista = col2.radio("Dendrogram type", ["Linear (interactive)", "Circular"], horizontal=True)

        Z = _calcular_linkage_cacheado(X_hca, metodo)

        mapa_color_clase = None
        if clases is not None:
            clases_unicas = list(pd.unique(como_texto(clases)))
            mapa_color_clase = {c: rgb_string_a_hex(CLASS_PALETTE[i % len(CLASS_PALETTE)])
                                 for i, c in enumerate(clases_unicas)}
            st.caption("Leaf labels are colored by class, so you can see at a glance whether "
                       "same-class samples cluster together (cluster purity).")
            leyenda = "  ".join(f"<span style='color:{color}'>■</span> {c}"
                                 for c, color in mapa_color_clase.items())
            st.markdown(leyenda, unsafe_allow_html=True)

        if vista.startswith("Linear"):
            fig = ff.create_dendrogram(X_hca, labels=list(ids), linkagefun=lambda x: Z)
            fig.update_layout(height=550 if mapa_color_clase is None else 610, title="Linear dendrogram")
            if mapa_color_clase is not None:
                orden_hojas = list(fig.layout.xaxis.ticktext)
                colores_hojas = [mapa_color_clase.get(str(clases[list(ids).index(hoja)]), "black")
                                 for hoja in orden_hojas]
                altura_max = float(Z[:, 2].max())
                y_marcador = -altura_max * 0.05
                fig.add_trace(go.Scatter(
                    x=list(fig.layout.xaxis.tickvals), y=[y_marcador] * len(orden_hojas),
                    mode="markers", marker=dict(size=9, color=colores_hojas, symbol="square"),
                    hoverinfo="skip", showlegend=False,
                ))
                fig.update_yaxes(range=[y_marcador * 1.8, altura_max * 1.05])
            st.plotly_chart(fig, width='stretch')
        else:
            id_a_clase = None
            if clases is not None:
                id_a_clase = {str(i): str(c) for i, c in zip(ids, clases)}
            fig_mpl = cu.dendrograma_circular_fig(Z, list(ids), id_a_clase=id_a_clase,
                                                    mapa_color_clase=mapa_color_clase)
            st.pyplot(fig_mpl)

# -----------------------------------------------------------------------
# TAB: OTHER TOOLS
# -----------------------------------------------------------------------
with tabs[6]:
    st.subheader("Other exploratory analysis tools")

    ids, X_activo, clases = datos_activos()
    numeros_onda = st.session_state.numeros_onda_pret
    X_hca = st.session_state.X_pret[indice_activo()]

    herramienta = st.selectbox(
        "Choose a tool",
        ["Loadings vs wavenumber", "Ranking of important variables",
         "Loadings correlation plot", "Mean spectrum by class",
         "Clustermap (heatmap + dendrogram)", "t-SNE", "UMAP", "MCR-ALS (mixture resolution)"],
        help="Pick which exploratory technique to run on the preprocessed data.",
    )

    tiene_pca = "cargas_completo" in st.session_state

    if herramienta in ["Loadings vs wavenumber", "Ranking of important variables",
                        "Loadings correlation plot"] and not tiene_pca:
        st.warning("Visit the PCA tab first.")

    elif herramienta == "Loadings vs wavenumber":
        cargas_completo = st.session_state.cargas_completo
        n_pcs = st.slider("How many PCs to show", 1, min(5, cargas_completo.shape[0]), 3,
                           help="Number of components to overlay on the loading plot, so you can "
                                "compare which wavenumbers drive each one.")
        fig = go.Figure()
        for i in range(n_pcs):
            fig.add_trace(go.Scatter(x=numeros_onda, y=cargas_completo[i], mode="lines", name=f"PC{i+1}"))
        fig.update_layout(height=450, xaxis_title="Wavenumber", yaxis_title="Loading")
        if numeros_onda[0] > numeros_onda[-1]:
            fig.update_xaxes(autorange="reversed")
        st.plotly_chart(fig, width='stretch')

    elif herramienta == "Ranking of important variables":
        cargas_completo = st.session_state.cargas_completo
        col1, col2 = st.columns(2)
        pc_elegido = col1.number_input("PC to analyze", min_value=1, max_value=cargas_completo.shape[0], value=1)
        top_n = col2.slider("How many variables to show", 5, 40, 15)
        idx_pc = pc_elegido - 1
        orden = np.argsort(np.abs(cargas_completo[idx_pc]))[::-1][:top_n]
        orden = orden[np.argsort(cargas_completo[idx_pc, orden])]
        colores = ["crimson" if v < 0 else "steelblue" for v in cargas_completo[idx_pc, orden]]
        fig = go.Figure(go.Bar(
            x=cargas_completo[idx_pc, orden],
            y=[f"{numeros_onda[i]:.0f}" for i in orden],
            orientation="h", marker_color=colores,
        ))
        fig.update_layout(height=max(350, 22 * top_n), xaxis_title=f"Loading on PC{pc_elegido}",
                           yaxis_title="Wavenumber")
        st.plotly_chart(fig, width='stretch')

    elif herramienta == "Loadings correlation plot":
        cargas_completo = st.session_state.cargas_completo
        autovalores = st.session_state.autovalores
        col1, col2, col3 = st.columns(3)
        pc_x = col1.number_input("PC on X axis", min_value=1, max_value=cargas_completo.shape[0], value=1)
        pc_y = col2.number_input("PC on Y axis", min_value=1, max_value=cargas_completo.shape[0], value=2)
        top_n = col3.slider("Variables to highlight", 5, 40, 15)

        load_x = cargas_completo[pc_x - 1] * np.sqrt(autovalores[pc_x - 1])
        load_y = cargas_completo[pc_y - 1] * np.sqrt(autovalores[pc_y - 1])
        escala = 1 / max(np.max(np.abs(load_x)), np.max(np.abs(load_y)))
        load_x_n, load_y_n = load_x * escala, load_y * escala
        peso = load_x_n ** 2 + load_y_n ** 2
        top_idx = np.argsort(peso)[-top_n:]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=load_x_n, y=load_y_n, mode="markers",
                                  marker=dict(size=6, color="steelblue", opacity=0.35),
                                  name="All variables"))
        fig.add_trace(go.Scatter(x=load_x_n[top_idx], y=load_y_n[top_idx], mode="markers+text",
                                  marker=dict(size=9, color="darkred"),
                                  text=[f"{numeros_onda[i]:.0f}" for i in top_idx],
                                  textposition="top center", name=f"Top {top_n}"))
        theta = np.linspace(0, 2 * np.pi, 100)
        for r in (0.5, 1.0):
            fig.add_trace(go.Scatter(x=r * np.cos(theta), y=r * np.sin(theta), mode="lines",
                                      line=dict(dash="dash", color="gray"), showlegend=False))
        fig.update_layout(height=600, xaxis_title=f"PC{pc_x}", yaxis_title=f"PC{pc_y}",
                           xaxis=dict(range=[-1.2, 1.2]), yaxis=dict(range=[-1.2, 1.2], scaleanchor="x"))
        st.plotly_chart(fig, width='stretch')

    elif herramienta == "Mean spectrum by class":
        if clases is None:
            st.warning("Define classes in the left-hand panel first.")
        else:
            fig = go.Figure()
            for c in np.unique(clases):
                mask = clases == c
                promedio = X_hca[mask].mean(axis=0)
                desvio = X_hca[mask].std(axis=0)
                fig.add_trace(go.Scatter(x=numeros_onda, y=promedio, mode="lines", name=str(c)))
                fig.add_trace(go.Scatter(
                    x=np.concatenate([numeros_onda, numeros_onda[::-1]]),
                    y=np.concatenate([promedio + desvio, (promedio - desvio)[::-1]]),
                    fill="toself", opacity=0.15, line=dict(width=0), showlegend=False,
                ))
            fig.update_layout(height=450, xaxis_title="Wavenumber", yaxis_title="Signal")
            if numeros_onda[0] > numeros_onda[-1]:
                fig.update_xaxes(autorange="reversed")
            st.plotly_chart(fig, width='stretch')

    elif herramienta == "Clustermap (heatmap + dendrogram)":
        df_heat = pd.DataFrame(X_hca, index=ids, columns=np.round(numeros_onda, 0))
        metodo_cm = st.selectbox("Linkage method", ["ward", "average", "complete", "single"], key="metodo_cm")
        fig_cm = sns.clustermap(df_heat, method=metodo_cm, metric="euclidean",
                                 col_cluster=False, cmap="viridis", figsize=(10, 7), xticklabels=False)
        st.pyplot(fig_cm.fig)

    elif herramienta == "t-SNE":
        from sklearn.manifold import TSNE
        perplejidad = st.slider("Perplexity", 5, min(50, max(6, X_hca.shape[0] - 1)), min(30, max(6, X_hca.shape[0] - 1)))
        if st.button("Compute t-SNE"):
            with st.spinner("Computing..."):
                emb = TSNE(n_components=2, perplexity=perplejidad, init="pca", random_state=0).fit_transform(X_hca)
            fig = px.scatter(x=emb[:, 0], y=emb[:, 1], hover_name=ids,
                              color=clases if clases is not None else None,
                color_discrete_sequence=CLASS_PALETTE,
                              labels={"x": "t-SNE 1", "y": "t-SNE 2"})
            fig.update_layout(height=500)
            st.plotly_chart(fig, width='stretch')

    elif herramienta == "UMAP":
        try:
            import umap
        except ImportError:
            st.error("The `umap-learn` package is missing (pip install umap-learn) for this option.")
        else:
            vecinos = st.slider("n_neighbors", 2, min(50, max(3, X_hca.shape[0] - 1)), min(15, max(3, X_hca.shape[0] - 1)))
            if st.button("Compute UMAP"):
                with st.spinner("Computing..."):
                    emb = umap.UMAP(n_components=2, n_neighbors=vecinos, random_state=0).fit_transform(X_hca)
                fig = px.scatter(x=emb[:, 0], y=emb[:, 1], hover_name=ids,
                                  color=clases if clases is not None else None,
                color_discrete_sequence=CLASS_PALETTE,
                                  labels={"x": "UMAP 1", "y": "UMAP 2"})
                fig.update_layout(height=500)
                st.plotly_chart(fig, width='stretch')

    elif herramienta == "MCR-ALS (mixture resolution)":
        st.caption("Multivariate Curve Resolution — Alternating Least Squares: decomposes your "
                   "spectra into a set of 'pure component' spectra and their concentration profile "
                   "across samples, without needing to know the pure spectra beforehand. Useful when "
                   "your samples are mixtures and you want to recover what the individual "
                   "constituents look like and how much of each is in every sample. Assumes "
                   "non-negative concentrations and spectra (the usual physical case).")
        n_componentes_mcr = st.slider(
            "Number of components to resolve", 2, min(8, X_hca.shape[0] - 1), 2,
            help="How many pure/underlying components MCR-ALS should try to recover. Too few won't "
                 "explain the mixtures well; too many risk splitting real signal into noise-fitting "
                 "components. Try comparing the lack-of-fit for a couple of values.",
        )
        if st.button("Run MCR-ALS", key="ejecutar_mcr"):
            with st.spinner("Running alternating least squares..."):
                resultado_mcr = cu.mcr_als(X_hca, n_componentes=n_componentes_mcr, max_iter=200)
            st.session_state["mcr_resultado"] = resultado_mcr
            st.session_state["mcr_n_componentes"] = n_componentes_mcr
        if "mcr_resultado" in st.session_state:
            resultado_mcr = st.session_state["mcr_resultado"]
            st.metric("Lack of fit", f"{resultado_mcr['lof_pct']:.2f}%",
                      help="Percentage of the data's variance NOT explained by the resolved "
                           "components — lower is better. Under ~5% is generally considered a good fit.")
            st.caption(f"Converged in {resultado_mcr['n_iter']} iterations.")

            fig_spectra_mcr = go.Figure()
            for k in range(resultado_mcr["S"].shape[0]):
                fig_spectra_mcr.add_trace(go.Scatter(x=numeros_onda, y=resultado_mcr["S"][k],
                                                       mode="lines", name=f"Component {k+1}"))
            fig_spectra_mcr.update_layout(height=400, title="Resolved pure-component spectra",
                                           xaxis_title="Wavenumber", yaxis_title="Signal (a.u.)")
            if numeros_onda[0] > numeros_onda[-1]:
                fig_spectra_mcr.update_xaxes(autorange="reversed")
            st.plotly_chart(fig_spectra_mcr, width='stretch')

            df_conc_mcr = pd.DataFrame(
                resultado_mcr["C"], index=ids,
                columns=[f"Component {k+1}" for k in range(resultado_mcr["C"].shape[1])],
            )
            fig_conc_mcr = go.Figure()
            for col in df_conc_mcr.columns:
                fig_conc_mcr.add_trace(go.Bar(x=df_conc_mcr.index.astype(str), y=df_conc_mcr[col], name=col))
            fig_conc_mcr.update_layout(height=400, barmode="stack", title="Relative concentration profile per sample",
                                        xaxis_title="Sample", yaxis_title="Relative concentration")
            st.plotly_chart(fig_conc_mcr, width='stretch')

            st.download_button(
                "⬇️ Download concentration profiles (Excel)",
                data=df_a_excel_bytes({"mcr_concentrations": df_conc_mcr.reset_index(names="id")}),
                file_name="mcr_als_concentrations.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="descargar_mcr_excel",
            )

# -----------------------------------------------------------------------
# TAB: CLASSIFICATION
# -----------------------------------------------------------------------
with tabs[7]:
    st.subheader("Supervised classification")
    if st.button("🔄 Reset this tab", key="reset_clf",
                 help="Clears all trained models, metrics, and plots from this tab, so you can "
                      "start a completely fresh run without any leftover results from before."):
        resetear_prefijo("clf_")
        st.rerun()

    ids_activos, X_activo_clf, clases_activas = datos_activos()
    X_modelado = st.session_state.X_pret[indice_activo()]
    eje_modelado = st.session_state.numeros_onda_pret

    if clases_activas is None:
        st.warning("Define classes in the left-hand panel (Data tab) to run a classification.")
    else:
        conteos = pd.Series(clases_activas).value_counts()
        if conteos.min() < 2:
            st.error(f"Some class has fewer than 2 samples ({conteos.to_dict()}). "
                     "At least 2 samples per class are needed to validate.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                modelos_elegidos = st.multiselect(
                    "Models to compare",
                    list(mu.crear_clasificadores().keys()),
                    default=["LDA", "PLS-DA", "Random Forest"],
                    help="Pick one or more algorithms to train and compare side by side in the same "
                         "run — the results table below will show every metric for each of them.",
                )
                metodo_seleccion = st.selectbox(
                    "Variable selection",
                    ["None", "Boruta", "Genetic Algorithm"],
                    help="Reduce the spectrum to just the most informative wavenumbers before "
                         "training. Boruta is conservative and keeps anything that helps at all; "
                         "the Genetic Algorithm explores combinations more freely but is slower and "
                         "a bit more random. 'None' uses the full spectrum.",
                )
            with col2:
                cv_folds_sel = st.selectbox(
                    "Cross-validation folds (internal)",
                    [3, 5, 10, "LOO (leave 1 sample out at a time)"], index=1,
                    help="How many folds to use for cross-validation. More folds = less biased but "
                         "slower and more variance between folds. LOO (Leave-One-Out) trains one "
                         "model per sample — most thorough, slowest, and typical for small datasets.",
                )
                cv_folds = "LOO" if cv_folds_sel == "LOO (leave 1 sample out at a time)" else cv_folds_sel
                if cv_folds == "LOO" and X_modelado.shape[0] > 150:
                    st.caption("⚠ LOO trains one model per sample — with many samples this can take a while.")
                prop_test = st.slider(
                    "Proportion for independent test set (0 = use everything for CV)",
                    0.0, 0.4, 0.2, step=0.05,
                    help="Fraction of samples set aside BEFORE training, never used to choose "
                         "anything — only to get one final, honest check of how the model performs "
                         "on data it has never influenced. 0 means skip this and use every sample "
                         "for cross-validation instead.",
                )
                metodo_split_sel = st.selectbox(
                    "Test set selection", ["Random (stratified)", "Kennard-Stone (representative)"],
                    disabled=(prop_test == 0),
                    help="How to choose which samples go into the test set. 'Random' picks them by "
                         "chance (stratified by class). 'Kennard-Stone' spreads the training set "
                         "across the spectral range within each class instead — every class still "
                         "ends up represented in both train and test.",
                )
                metodo_split = "kennard_stone" if metodo_split_sel.startswith("Kennard-Stone") else "random"
                if metodo_split == "kennard_stone":
                    st.caption("ℹ️ Kennard-Stone picks the TRAINING set to broadly cover the spectral "
                               "space within EACH class separately (it will include the more 'extreme' "
                               "samples of each class); the remaining, more 'typical' samples of each "
                               "class become the test set — so every class is guaranteed to appear in "
                               "both.")
                optimizar = st.checkbox(
                    "Optimize hyperparameters (slower)",
                    help="Automatically search for better settings for each model (e.g. how many "
                         "trees in a Random Forest) instead of using the defaults. Usually improves "
                         "results a bit, at the cost of extra computation time.",
                )
                preset_opt = st.selectbox(
                    "Optimization budget", list(PRESETS_OPTIMIZACION.keys()), index=1, disabled=not optimizar,
                    help="How much effort to spend searching for good hyperparameters — see the "
                         "explanation below once you pick one.",
                )
                metodo_opt = PRESETS_OPTIMIZACION[preset_opt][0]

            if optimizar:
                st.caption(f"ℹ️ **{preset_opt}**: {PRESETS_OPTIMIZACION[preset_opt][1]}")

            if optimizar and any(m in mu.ALGORITMOS_LENTOS_AL_OPTIMIZAR for m in modelos_elegidos):
                lentos_elegidos = [m for m in modelos_elegidos if m in mu.ALGORITMOS_LENTOS_AL_OPTIMIZAR]
                extra = " With LOO this can take several minutes." if cv_folds == "LOO" else ""
                st.caption(f"⚠ Optimizing hyperparameters for {', '.join(lentos_elegidos)} can still take a while "
                           f"on a shared/limited CPU. If it's too slow, try 'Fast', or fewer folds.{extra}")

            if optimizar and len(modelos_elegidos) > 3:
                st.warning(f"⚠️ You're optimizing hyperparameters for {len(modelos_elegidos)} models at once. "
                           "Each one runs its own search in parallel across CPU cores, so doing several "
                           "together can exhaust the machine's memory and cause the app to crash or freeze. "
                           "We recommend optimizing **up to 3 models at a time** — train the rest with "
                           "default settings first, then optimize the best candidates separately.")

            if metodo_seleccion == "Boruta":
                c1, c2 = st.columns(2)
                boruta_max_iter = c1.slider(
                    "Boruta iterations", 10, 500, 40, step=10, key="boruta_iter_clf",
                    help="How many comparison rounds to run against the random 'shadow' variables. "
                         "More iterations give a more stable decision but take longer.",
                )
                boruta_alpha = c2.select_slider(
                    "Boruta p-value (alpha)", options=[0.01, 0.02, 0.05, 0.1], value=0.05, key="boruta_alpha_clf",
                    help="Significance level for keeping a variable: smaller (e.g. 0.01) is stricter "
                         "and keeps fewer, more confidently relevant variables; larger (e.g. 0.1) is "
                         "more permissive.",
                )
            if metodo_seleccion == "Genetic Algorithm":
                c1, c2 = st.columns(2)
                ga_poblacion = c1.slider("Population size", 10, 60, 20, key="ga_pob_clf")
                ga_generaciones = c2.slider("Generations", 5, 50, 15, key="ga_gen_clf")

            if st.session_state.get("clf_resultados") is not None:
                insignia_estado("clf_firma", "This trained classification model")

            if st.button("🚀 Train and evaluate (Classification)", disabled=len(modelos_elegidos) == 0):
                # Clear secondary results tied to the PREVIOUS set of trained models (a
                # statistical comparison or learning curve computed for models A/B/C would
                # otherwise linger on screen after retraining with a different D/E/F).
                for _clave in ["clf_pvalores", "clf_puntajes_cv", "clf_comparacion_metodo",
                                "clf_curva_aprendizaje", "clf_curva_modelo", "clf_ultima_ficha"]:
                    st.session_state.pop(_clave, None)
                mascara_variables = None
                _msg_espera = "This may take a few minutes..." if (optimizar or metodo_seleccion in ("Boruta", "Genetic Algorithm")) else "Training..."
                with st.spinner(f"Selecting variables ({_msg_espera})" if metodo_seleccion != "None" else _msg_espera):
                    if metodo_seleccion == "Boruta":
                        try:
                            mascara_variables = mu.seleccionar_variables_boruta(
                                X_modelado, clases_activas, es_clasificacion=True,
                                max_iter=boruta_max_iter, alpha=boruta_alpha,
                            )
                            if mascara_variables.sum() == 0:
                                st.warning("Boruta did not select any variable; using all of them.")
                                mascara_variables = None
                        except Exception as e:
                            st.error(f"Boruta failed ({e}); using all variables.")
                    elif metodo_seleccion == "Genetic Algorithm":
                        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
                        cv_ga = 3 if cv_folds == "LOO" else min(3, cv_folds)
                        ga = mu.SeleccionGenetica(
                            LinearDiscriminantAnalysis(), es_clasificacion=True,
                            tam_poblacion=ga_poblacion, n_generaciones=ga_generaciones,
                            cv=cv_ga, random_state=0,
                        )
                        ga.fit(X_modelado, clases_activas)
                        mascara_variables = ga.mejor_mascara_

                    X_sel = X_modelado[:, mascara_variables] if mascara_variables is not None else X_modelado

                    resultados = {}
                    hiperparametros_optimos = {}
                    descripcion_opt_usada = {}
                    catalogo = mu.crear_clasificadores()
                    for nombre in modelos_elegidos:
                        modelo = catalogo[nombre]
                        try:
                            if optimizar and nombre in mu.GRILLAS_CLASIFICACION:
                                modelo, mejores_params, _, desc_opt = mu.optimizar_hiperparametros(
                                    modelo, mu.GRILLAS_CLASIFICACION[nombre], X_sel, clases_activas,
                                    es_clasificacion=True, cv=cv_folds, metodo=metodo_opt,
                                )
                                hiperparametros_optimos[nombre] = mejores_params
                                descripcion_opt_usada[nombre] = desc_opt
                            resultados[nombre] = mu.entrenar_evaluar_clasificacion(
                                modelo, X_sel, clases_activas, ids=ids_activos,
                                cv=cv_folds, proporcion_test=prop_test, metodo_split=metodo_split,
                            )
                        except Exception as e:
                            resultados[nombre] = {"error": str(e)}

                st.session_state["clf_resultados"] = resultados
                st.session_state["clf_mascara_variables"] = mascara_variables
                st.session_state["clf_descripcion_opt"] = descripcion_opt_usada
                st.session_state["clf_ids_usados"] = ids_activos
                st.session_state["clf_eje_usado"] = eje_modelado
                st.session_state["clf_pasos_pretratamiento"] = st.session_state.pasos_pretratamiento
                st.session_state["clf_hiperparametros"] = hiperparametros_optimos
                st.session_state["clf_espectro_promedio"] = X_modelado.mean(axis=0)
                st.session_state["clf_cv_folds"] = cv_folds
                st.session_state["clf_prop_test"] = prop_test
                st.session_state["clf_metodo_split"] = metodo_split_sel
                st.session_state["clf_metodo_seleccion"] = metodo_seleccion
                st.session_state["clf_n_muestras"] = X_modelado.shape[0]
                st.session_state["clf_firma"] = firma_datos_activos()

            if "clf_resultados" in st.session_state:
                resultados = st.session_state["clf_resultados"]
                mascara_variables = st.session_state["clf_mascara_variables"]
                hiperparametros_optimos = st.session_state.get("clf_hiperparametros", {})
                descripcion_opt_usada = st.session_state.get("clf_descripcion_opt", {})

                if mascara_variables is not None:
                    st.caption(f"Selected variables: {int(mascara_variables.sum())} of {len(mascara_variables)}.")
                    with st.expander("📍 View selected variables on the mean spectrum"):
                        fig_vars = go.Figure()
                        eje_r = st.session_state["clf_eje_usado"]
                        fig_vars.add_trace(go.Scatter(x=eje_r, y=st.session_state["clf_espectro_promedio"],
                                                       mode="lines", line=dict(color="#5F5E5A"), name="Mean spectrum"))
                        idx_sel = np.where(mascara_variables)[0]
                        bloques = np.split(idx_sel, np.where(np.diff(idx_sel) != 1)[0] + 1) if len(idx_sel) else []
                        for j, bloque in enumerate(bloques):
                            x0, x1 = eje_r[bloque[0]], eje_r[bloque[-1]]
                            fig_vars.add_vrect(x0=min(x0, x1), x1=max(x0, x1), fillcolor="#0F6E56",
                                                opacity=0.25, line_width=0)
                        fig_vars.update_layout(height=380, xaxis_title="Wavenumber", yaxis_title="Signal")
                        if eje_r[0] > eje_r[-1]:
                            fig_vars.update_xaxes(autorange="reversed")
                        st.plotly_chart(fig_vars, width='stretch')

                        st.markdown("**List of selected variables**")
                        df_vars_sel = df_variables_seleccionadas(eje_r, mascara_variables)
                        st.dataframe(df_vars_sel, width='stretch', height=200)
                        st.download_button(
                            "⬇️ Download selected variables (Excel)",
                            data=df_a_excel_bytes({"selected_variables": df_vars_sel}),
                            file_name="selected_variables_classification.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="descargar_vars_clf",
                        )

                filas = []
                for nombre, res in resultados.items():
                    if "error" in res:
                        filas.append({"Model": nombre, "Error": res["error"]})
                        continue
                    fila = {
                        "Model": nombre,
                        "Accuracy (CV)": round(res["cv"]["accuracy"], 3),
                        "Bal. Accuracy (CV)": round(res["cv"]["balanced_accuracy"], 3),
                        "Sensitivity (CV)": round(res["cv"]["sensibilidad_macro"], 3),
                        "Specificity (CV)": round(res["cv"]["especificidad_macro"], 3),
                        "F1 macro (CV)": round(res["cv"]["f1_macro"], 3),
                        "Kappa (CV)": round(res["cv"]["kappa"], 3),
                        "MCC (CV)": round(res["cv"]["mcc"], 3),
                        "AUC (CV)": round(res["cv"]["auc"], 3) if res["cv"]["auc"] is not None else "n/a",
                    }
                    if "test" in res:
                        fila["Accuracy (test)"] = round(res["test"]["accuracy"], 3)
                        fila["Sensitivity (test)"] = round(res["test"]["sensibilidad_macro"], 3)
                        fila["Specificity (test)"] = round(res["test"]["especificidad_macro"], 3)
                        fila["Kappa (test)"] = round(res["test"]["kappa"], 3)
                        fila["MCC (test)"] = round(res["test"]["mcc"], 3)
                        fila["AUC (test)"] = round(res["test"]["auc"], 3) if res["test"]["auc"] is not None else "n/a"
                    filas.append(fila)
                df_metricas_clf = pd.DataFrame(filas)
                st.dataframe(df_metricas_clf, width='stretch')
                st.download_button(
                    "⬇️ Download metrics table (Excel)",
                    data=df_a_excel_bytes({"classification_metrics": df_metricas_clf}),
                    file_name="classification_metrics.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="descargar_metricas_clf",
                )

                modelos_exitosos = [n for n in resultados if "error" not in resultados[n]]
                if len(modelos_exitosos) >= 2:
                    hay_test_clf = "test" in resultados[modelos_exitosos[0]]
                    with st.expander("📊 Compare models statistically"):
                        if hay_test_clf:
                            st.caption("Compares models on the held-out TEST set predictions, sample by "
                                       "sample (McNemar's test) — this is what can actually reveal "
                                       "overfitting, since two models can look equally good in "
                                       "cross-validation yet behave very differently on genuinely unseen "
                                       "data. A p-value below 0.05 means the difference is unlikely to be "
                                       "just chance.")
                        else:
                            st.caption("No independent test set was configured, so this compares models "
                                       "using a paired t-test on their per-fold cross-validation scores "
                                       "instead (same folds for every model). Note this can't detect "
                                       "overfitting the way a held-out test set would — consider setting "
                                       "aside a test proportion above 0% for a more trustworthy comparison.")
                        if st.button("Run comparison", key="comparar_modelos_clf"):
                            if hay_test_clf:
                                pvalores, puntajes = mu.comparar_modelos_en_test(
                                    {n: resultados[n] for n in modelos_exitosos}, es_clasificacion=True,
                                )
                            else:
                                mascara_cmp = st.session_state["clf_mascara_variables"]
                                # Use the exact same train-only subset the main CV metrics used, so the
                                # held-out test set (if any) never leaks into model comparison.
                                ids_train_ref = resultados[modelos_exitosos[0]]["cv"]["ids"]
                                mask_train_cmp = np.isin(ids_activos, ids_train_ref)
                                X_cmp = X_modelado[mask_train_cmp]
                                y_cmp = clases_activas[mask_train_cmp]
                                if mascara_cmp is not None:
                                    X_cmp = X_cmp[:, mascara_cmp]
                                modelos_para_comparar = {
                                    n: mu.clone(resultados[n]["modelo_final"]) for n in modelos_exitosos
                                }
                                with st.spinner("Running cross-validation for every model on the same folds..."):
                                    pvalores, puntajes = mu.comparar_modelos_estadisticamente(
                                        modelos_para_comparar, X_cmp, y_cmp, es_clasificacion=True,
                                        cv=st.session_state["clf_cv_folds"],
                                    )
                            st.session_state["clf_pvalores"] = pvalores
                            st.session_state["clf_puntajes_cv"] = puntajes
                            st.session_state["clf_comparacion_metodo"] = (
                                "McNemar's test on the held-out test set" if hay_test_clf
                                else "Paired t-test on cross-validation folds (no test set configured)"
                            )
                        if "clf_pvalores" in st.session_state:
                            interpretacion = mu.interpretar_comparacion_modelos(
                                st.session_state["clf_pvalores"], st.session_state["clf_puntajes_cv"],
                            )
                            if interpretacion["hay_diferencias"]:
                                for msg in interpretacion["mensajes"]:
                                    st.success(msg)
                            else:
                                st.info(interpretacion["mensajes"][0])
                            st.markdown("**p-values:**")
                            st.dataframe(st.session_state["clf_pvalores"].style.format("{:.4f}"), width='stretch')

                nombre_detalle = st.selectbox(
                    "View details for", [n for n in resultados if "error" not in resultados[n]],
                    help="Choose which of the trained models to inspect below: confusion matrix, "
                         "hyperparameters, learning curve, and the report/save options.",
                )
                if nombre_detalle:
                    res = resultados[nombre_detalle]

                    if hiperparametros_optimos.get(nombre_detalle):
                        with st.container(border=True):
                            st.markdown(f"**⚙️ Optimized hyperparameters — {nombre_detalle}**")
                            cols_hp = st.columns(len(hiperparametros_optimos[nombre_detalle]))
                            for col_hp, (k, v) in zip(cols_hp, hiperparametros_optimos[nombre_detalle].items()):
                                col_hp.metric(k, str(v))
                            st.caption(f"🔧 Method: {descripcion_opt_usada.get(nombre_detalle, 'n/a')}")

                    fuente = st.radio("Confusion matrix on:", ["CV", "Test"] if "test" in res else ["CV"],
                                       horizontal=True, key="fuente_matriz_clf",
                                       help="'CV' shows the cross-validation predictions; 'Test' shows "
                                            "the independent held-out test set, if one was configured.")
                    datos_matriz = res["cv"] if fuente == "CV" else res["test"]
                    clases_orden = res["cv"]["clases"]
                    if fuente == "Test":
                        y_t, y_p = res["test"]["y_true"], res["test"]["y_pred"]
                        matriz = mu.confusion_matrix(y_t, y_p, labels=clases_orden)
                    else:
                        matriz = res["cv"]["matriz_confusion"]

                    fig_cm = px.imshow(matriz, text_auto=True, x=list(clases_orden), y=list(clases_orden),
                                        labels=dict(x="Predicted", y="Actual", color="Count"),
                                        color_continuous_scale="Blues")
                    fig_cm.update_layout(height=420, title=f"Confusion matrix — {nombre_detalle} ({fuente})")
                    st.plotly_chart(fig_cm, width='stretch')

                    ids_fuente = res["cv"]["ids"] if fuente == "CV" else res["test"]["ids"]
                    y_true_fuente = res["cv"]["y_true"] if fuente == "CV" else res["test"]["y_true"]
                    y_pred_fuente = res["cv"]["y_pred"] if fuente == "CV" else res["test"]["y_pred"]
                    df_exp = mu.exportar_predicciones_clasificacion(ids_fuente, y_true_fuente, y_pred_fuente)
                    st.download_button(
                        f"⬇️ Download predictions ({nombre_detalle}, {fuente})",
                        data=df_exp.to_csv(index=False).encode("utf-8"),
                        file_name=f"predictions_{nombre_detalle}_{fuente}.csv", mime="text/csv",
                    )

                    with st.expander(f"📈 Learning curve — {nombre_detalle}"):
                        st.caption("Shows performance vs. how many training samples were used. If both "
                                   "curves are still rising and far apart, more samples would likely help. "
                                   "If they're flat and close together, the model has plateaued.")
                        if st.button("Compute learning curve", key="curva_aprendizaje_clf"):
                            mascara_lc = st.session_state["clf_mascara_variables"]
                            X_lc = X_modelado[:, mascara_lc] if mascara_lc is not None else X_modelado
                            with st.spinner("Training with increasing sample sizes (this may take a few minutes)..."):
                                curva = mu.calcular_curva_aprendizaje(
                                    mu.clone(res["modelo_final"]), X_lc, clases_activas, es_clasificacion=True,
                                    cv=min(5, st.session_state["clf_cv_folds"]) if st.session_state["clf_cv_folds"] != "LOO" else 5,
                                )
                            st.session_state["clf_curva_aprendizaje"] = curva
                            st.session_state["clf_curva_modelo"] = nombre_detalle
                        if st.session_state.get("clf_curva_modelo") == nombre_detalle and "clf_curva_aprendizaje" in st.session_state:
                            curva = st.session_state["clf_curva_aprendizaje"]
                            fig_lc = go.Figure()
                            fig_lc.add_trace(go.Scatter(x=curva["train_sizes"], y=curva["train_scores_mean"],
                                                         mode="lines+markers", name="Training score",
                                                         line=dict(color="#185FA5")))
                            fig_lc.add_trace(go.Scatter(x=curva["train_sizes"], y=curva["val_scores_mean"],
                                                         mode="lines+markers", name="Cross-validation score",
                                                         line=dict(color="#B91C1C")))
                            fig_lc.update_layout(height=380, xaxis_title="Training set size (samples)",
                                                  yaxis_title=curva["scoring"])
                            st.plotly_chart(fig_lc, width='stretch')

                    st.divider()
                    st.markdown("**📄 Report for this analysis**")
                    st.caption("Includes: dataset used, spectra, preprocessing, selected variables, "
                               "a comparison table of all trained models, and the detail "
                               f"({', '.join([nombre_detalle])}) with confusion matrix and hyperparameters.")
                    if st.button("🖨️ Generate report (PDF)", key="generar_reporte_clf"):
                        secciones_rep = [
                            {"tipo": "titulo", "texto": "1. Dataset"},
                            {"tipo": "clave_valor", "pares": [
                                ("Samples used", int(X_modelado.shape[0])),
                                ("Original variables", int(X_modelado.shape[1])),
                                ("Class distribution", ", ".join(
                                    f"{c}: {n}" for c, n in pd.Series(clases_activas).value_counts().items())),
                                ("Preprocessing", " -> ".join(
                                    p[0] for p in st.session_state["clf_pasos_pretratamiento"]) or "none"),
                            ]},
                            {"tipo": "imagen", "fig": ru.fig_espectros(
                                eje_modelado, X_modelado, clases=como_texto(clases_activas), titulo="Preprocessed spectra")},
                            {"tipo": "titulo", "texto": "2. Variable selection"},
                            {"tipo": "clave_valor", "pares": [
                                ("Method", st.session_state["clf_metodo_seleccion"]),
                                ("Selected variables", int(mascara_variables.sum()) if mascara_variables is not None
                                 else f"all ({X_modelado.shape[1]})"),
                            ]},
                        ]
                        if mascara_variables is not None:
                            secciones_rep.append({"tipo": "imagen", "fig": ru.fig_variables_seleccionadas(
                                eje_modelado, X_modelado.mean(axis=0), mascara_variables)})
                            idx_sel_rep = np.where(mascara_variables)[0]
                            secciones_rep.append({"tipo": "tabla",
                                "encabezados": ["Position", "Wavenumber"],
                                "filas": [[str(i), f"{eje_modelado[i]:.2f}"] for i in idx_sel_rep[:60]],
                                "anchos": [30, 70]})
                            if len(idx_sel_rep) > 60:
                                secciones_rep.append({"tipo": "parrafo",
                                    "texto": f"(showing the first 60 of {len(idx_sel_rep)} variables — "
                                             "the full list is in the Excel file downloadable from the app)"})
                        secciones_rep += [
                            {"tipo": "salto_pagina"},
                            {"tipo": "titulo", "texto": "3. Model comparison"},
                            {"tipo": "clave_valor", "pares": [
                                ("Cross-validation", f"{res['cv']['cv_folds_usados']}-fold" if cv_folds != "LOO"
                                 else f"Leave-One-Out ({res['cv']['cv_folds_usados']} repetitions)"),
                                ("Test proportion", f"{int(prop_test * 100)}% ({metodo_split_sel})"),
                            ]},
                            {"tipo": "tabla",
                             "encabezados": list(df_metricas_clf.columns),
                             "filas": df_metricas_clf.astype(str).values.tolist()},
                            {"tipo": "titulo", "texto": f"4. Detail: {nombre_detalle}"},
                            {"tipo": "imagen", "fig": ru.fig_matriz_confusion(matriz, list(clases_orden))},
                        ]
                        if hiperparametros_optimos.get(nombre_detalle):
                            secciones_rep.append({"tipo": "subtitulo", "texto": "Optimal hyperparameters"})
                            secciones_rep.append({"tipo": "clave_valor",
                                "pares": [("Optimization method", descripcion_opt_usada.get(nombre_detalle, "n/a"))]
                                + list(hiperparametros_optimos[nombre_detalle].items())})
                        if st.session_state.get("clf_curva_modelo") == nombre_detalle and "clf_curva_aprendizaje" in st.session_state:
                            secciones_rep.append({"tipo": "subtitulo", "texto": f"Learning curve — {nombre_detalle}"})
                            secciones_rep.append({"tipo": "imagen",
                                "fig": ru.fig_curva_aprendizaje(st.session_state["clf_curva_aprendizaje"], nombre_detalle)})
                        if "clf_pvalores" in st.session_state:
                            secciones_rep.append({"tipo": "salto_pagina"})
                            secciones_rep.append({"tipo": "titulo", "texto": "5. Statistical comparison between models"})
                            secciones_rep.append({"tipo": "parrafo",
                                "texto": f"Method: {st.session_state.get('clf_comparacion_metodo', 'n/a')}. "
                                         "A p-value below 0.05 suggests the performance difference between "
                                         "that pair of models is unlikely to be due to chance alone."})
                            interpretacion_rep = mu.interpretar_comparacion_modelos(
                                st.session_state["clf_pvalores"], st.session_state["clf_puntajes_cv"],
                            )
                            for msg in interpretacion_rep["mensajes"]:
                                secciones_rep.append({"tipo": "parrafo", "texto": msg.replace("**", "")})
                            secciones_rep.append({"tipo": "imagen",
                                "fig": ru.fig_comparacion_pvalores(st.session_state["clf_pvalores"])})

                        pdf_reporte_clf = ru.generar_reporte(
                            "Classification Report", f"Models: {', '.join(modelos_elegidos)}", secciones_rep,
                        )
                        st.download_button(
                            "⬇️ Download full report (PDF)",
                            data=bytes(pdf_reporte_clf.output()),
                            file_name="classification_report.pdf", mime="application/pdf",
                            key="descargar_reporte_clf",
                        )
                    st.divider()

                    nombre_guardado = st.text_input("Name to save this model as", value=nombre_detalle)
                    if st.button("💾 Save this model for the Prediction tab"):
                        id_trazabilidad = mu.generar_id_trazabilidad()
                        if clases_activas is not None:
                            desc_y = ", ".join(f"{c}: {n}" for c, n in pd.Series(clases_activas).value_counts().items())
                        else:
                            desc_y = "n/a"
                        cv_desc = (f"Leave-One-Out ({res['cv']['cv_folds_usados']} repetitions)"
                                   if cv_folds == "LOO" else f"{res['cv']['cv_folds_usados']}-fold")
                        metricas_ficha = {
                            "Accuracy (CV)": round(res["cv"]["accuracy"], 3),
                            "Balanced Accuracy (CV)": round(res["cv"]["balanced_accuracy"], 3),
                            "Sensitivity (CV)": round(res["cv"]["sensibilidad_macro"], 3),
                            "Specificity (CV)": round(res["cv"]["especificidad_macro"], 3),
                            "F1 macro (CV)": round(res["cv"]["f1_macro"], 3),
                            "Kappa (CV)": round(res["cv"]["kappa"], 3),
                            "MCC (CV)": round(res["cv"]["mcc"], 3),
                            "AUC (CV)": round(res["cv"]["auc"], 3) if res["cv"]["auc"] is not None else "n/a",
                        }
                        if "test" in res:
                            metricas_ficha.update({
                                "Accuracy (test)": round(res["test"]["accuracy"], 3),
                                "Sensitivity (test)": round(res["test"]["sensibilidad_macro"], 3),
                                "Specificity (test)": round(res["test"]["especificidad_macro"], 3),
                                "Kappa (test)": round(res["test"]["kappa"], 3),
                                "MCC (test)": round(res["test"]["mcc"], 3),
                                "AUC (test)": round(res["test"]["auc"], 3) if res["test"]["auc"] is not None else "n/a",
                            })
                        ficha = {
                            "id_trazabilidad": id_trazabilidad,
                            "fecha_creacion": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "nombre_modelo": nombre_guardado,
                            "tipo": "classification",
                            "algoritmo": nombre_detalle,
                            "n_muestras": int(X_modelado.shape[0]),
                            "n_variables_totales": int(X_modelado.shape[1]),
                            "n_variables_usadas": int(mascara_variables.sum()) if mascara_variables is not None else int(X_modelado.shape[1]),
                            "descripcion_y": desc_y,
                            "pretratamiento_desc": " -> ".join(p[0] for p in st.session_state["clf_pasos_pretratamiento"]) or "none",
                            "seleccion_variables_desc": st.session_state["clf_metodo_seleccion"],
                            "hiperparametros": hiperparametros_optimos.get(nombre_detalle, {}),
                            "metodo_optimizacion": descripcion_opt_usada.get(nombre_detalle, "none"),
                            "cv_desc": cv_desc,
                            "prop_test_desc": f"{int(st.session_state['clf_prop_test'] * 100)}% ({st.session_state.get('clf_metodo_split', 'Random')})",
                            "metricas": metricas_ficha,
                            "entorno_software": mu.info_entorno_software(),
                        }
                        st.session_state.modelos_guardados[nombre_guardado] = {
                            "tipo": "classification",
                            "modelo": res["modelo_final"],
                            "mascara_variables": mascara_variables,
                            "numeros_onda": st.session_state["clf_eje_usado"],
                            "pasos_pretratamiento": st.session_state["clf_pasos_pretratamiento"],
                            "ficha": ficha,
                        }
                        st.session_state["clf_ultima_ficha"] = ficha
                        st.success(f"Model '{nombre_guardado}' saved (traceability ID: {id_trazabilidad}). "
                                   "It is now available in the Prediction tab.")

                    if st.session_state.get("clf_ultima_ficha", {}).get("nombre_modelo") == nombre_guardado:
                        pdf_ficha = ru.generar_pdf_ficha_modelo(
                            st.session_state["clf_ultima_ficha"],
                            fig_extra=ru.fig_matriz_confusion(matriz, list(clases_orden)),
                            titulo_fig_extra="Confusion matrix",
                        )
                        st.download_button(
                            "📄 Download model card (PDF)",
                            data=bytes(pdf_ficha.output()),
                            file_name=f"model_card_{nombre_guardado}.pdf", mime="application/pdf",
                        )

# TAB: SIMCA
# -----------------------------------------------------------------------
with tabs[8]:
    st.subheader("SIMCA — Soft Independent Modeling of Class Analogies")
    if st.button("🔄 Reset this tab", key="reset_simca",
                 help="Clears all trained SIMCA models and results from this tab, so you can start "
                      "a completely fresh run without any leftover results from before."):
        resetear_prefijo("simca_")
        st.rerun()
    st.caption("Unlike LDA/PLS-DA/Random Forest (which always force a pick among the trained "
               "classes), SIMCA builds one PCA model PER CLASS and asks, independently for each "
               "one, 'does this sample look like a member of this class?'. A sample can end up "
               "accepted by exactly one class (a clean call), by several (an ambiguous / "
               "overlapping region), or by none (a genuine outlier — possibly a class the model "
               "has never seen). This makes it a natural fit for authenticity / quality-control "
               "questions such as 'is this really wine type B, or something else entirely?'.")

    ids_simca, _, clases_simca = datos_activos()
    X_modelado_simca = st.session_state.X_pret[indice_activo()]
    eje_modelado_simca = st.session_state.numeros_onda_pret

    if clases_simca is None:
        st.warning("Define classes in the left-hand panel (Data tab) to build SIMCA models.")
    else:
        conteos_simca = pd.Series(clases_simca).value_counts()
        if conteos_simca.min() < 4:
            st.error(f"Some class has fewer than 4 samples ({conteos_simca.to_dict()}). SIMCA needs "
                     "a handful of samples per class to fit a meaningful class-specific PCA model.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                varianza_objetivo_simca = st.slider(
                    "Target variance per class model", 0.80, 0.99, 0.90, step=0.01,
                    help="Each class gets its own PCA with just enough components to reach this "
                         "cumulative explained variance — capped at roughly a third of that class's "
                         "calibration samples, so the model doesn't run out of residual degrees of "
                         "freedom (which would make it fit calibration noise instead of real "
                         "structure, and fail to generalize to new samples).",
                )
                alpha_simca = st.slider("Significance level (alpha)", 0.01, 0.10, 0.05, step=0.01,
                                         key="alpha_simca",
                                         help="Controls how strict each class boundary is. Smaller "
                                              "alpha (e.g. 0.01) = a wider, more permissive boundary; "
                                              "larger alpha (e.g. 0.10) = a tighter one that rejects "
                                              "more borderline samples. 0.05 is the usual default.")
            with col2:
                prop_test_simca = st.slider(
                    "Proportion for independent test set (0 = validate on calibration only)",
                    0.0, 0.4, 0.3, step=0.05, key="test_simca",
                    help="Fraction of each class's samples set aside to check whether the class "
                         "model actually accepts genuinely new samples of that class — leaving this "
                         "at 0 only checks how well the model fits the same data it was built from, "
                         "which is optimistic.",
                )
                metodo_split_sel_simca = st.selectbox(
                    "Test set selection", ["Random (stratified)", "Kennard-Stone (representative)"],
                    disabled=(prop_test_simca == 0), key="split_simca",
                    help="How to choose which samples go into the test set for each class. 'Random' "
                         "picks them by chance; 'Kennard-Stone' spreads the calibration set across "
                         "that class's spectral range instead.",
                )
                metodo_split_simca = "kennard_stone" if metodo_split_sel_simca.startswith("Kennard-Stone") else "random"

            if st.button("🚀 Train SIMCA models (one per class)"):
                clases_unicas_simca = np.unique(clases_simca)
                if prop_test_simca > 0:
                    idx_train_s, idx_test_s = mu.dividir_train_test(
                        X_modelado_simca, clases_simca, ids_simca, prop_test_simca,
                        es_clasificacion=True, metodo_split=metodo_split_simca,
                    )
                else:
                    idx_train_s, idx_test_s = np.arange(len(clases_simca)), np.arange(len(clases_simca))

                modelos_simca = {}
                errores_simca = {}
                with st.spinner("Fitting one PCA model per class (this may take a few minutes)..."):
                    for c in clases_unicas_simca:
                        mask_c_train = (clases_simca[idx_train_s] == c)
                        X_c = X_modelado_simca[idx_train_s][mask_c_train]
                        try:
                            modelos_simca[c] = cu.entrenar_modelo_simca(
                                X_c, varianza_objetivo=varianza_objetivo_simca, alpha=alpha_simca,
                            )
                        except Exception as e:
                            errores_simca[c] = str(e)

                # Evaluate every TEST sample (regardless of its true class) against every class model
                X_eval = X_modelado_simca[idx_test_s]
                clases_eval_true = clases_simca[idx_test_s]
                ids_eval = ids_simca[idx_test_s]
                resultados_por_clase = {}
                matriz_dentro = pd.DataFrame(index=ids_eval, columns=list(modelos_simca.keys()), dtype=bool)
                for c, modelo_c in modelos_simca.items():
                    T2_c, Q_c, dentro_c = cu.evaluar_muestras_simca(X_eval, modelo_c)
                    resultados_por_clase[c] = {"T2": T2_c, "Q": Q_c, "dentro": dentro_c}
                    matriz_dentro[c] = dentro_c

                n_clases_aceptado = matriz_dentro.sum(axis=1)
                st.session_state["simca_modelos"] = modelos_simca
                st.session_state["simca_errores"] = errores_simca
                st.session_state["simca_resultados_por_clase"] = resultados_por_clase
                st.session_state["simca_matriz_dentro"] = matriz_dentro
                st.session_state["simca_clases_eval_true"] = clases_eval_true
                st.session_state["simca_ids_eval"] = ids_eval
                st.session_state["simca_n_clases_aceptado"] = n_clases_aceptado
                st.session_state["simca_eje_usado"] = eje_modelado_simca
                st.session_state["simca_pasos_pretratamiento"] = st.session_state.pasos_pretratamiento
                st.session_state["simca_prop_test"] = prop_test_simca
                st.session_state["simca_metodo_split"] = metodo_split_sel_simca
                st.session_state["simca_alpha"] = alpha_simca
                st.session_state["simca_varianza_objetivo"] = varianza_objetivo_simca
                st.session_state["simca_n_muestras"] = int(X_modelado_simca.shape[0])
                st.session_state["simca_validacion_es_calibracion"] = (prop_test_simca == 0)

            if "simca_modelos" in st.session_state:
                modelos_simca = st.session_state["simca_modelos"]
                errores_simca = st.session_state["simca_errores"]
                resultados_por_clase = st.session_state["simca_resultados_por_clase"]
                matriz_dentro = st.session_state["simca_matriz_dentro"]
                clases_eval_true = st.session_state["simca_clases_eval_true"]
                n_clases_aceptado = st.session_state["simca_n_clases_aceptado"]

                if errores_simca:
                    for c, err in errores_simca.items():
                        st.error(f"Class '{c}': {err}")

                clases_limitadas = [c for c, m in modelos_simca.items() if m.get("limitado_por_muestras")]
                if clases_limitadas:
                    st.info(f"ℹ️ For class(es) {', '.join(clases_limitadas)}, the target variance could not "
                            "be reached without using too many components for the available calibration "
                            "samples — the number of components was capped for stability. Consider adding "
                            "more calibration samples for these classes, or lowering the target variance.")

                if st.session_state["simca_validacion_es_calibracion"]:
                    st.caption("⚠ No independent test set was used — the stats below are computed on "
                               "the same samples used to fit each model (optimistic; only a rough check).")

                st.markdown("**Per-class model summary**")
                filas_simca = []
                for c, modelo_c in modelos_simca.items():
                    es_miembro = (clases_eval_true == c)
                    dentro_c = matriz_dentro[c].to_numpy()
                    sensibilidad = dentro_c[es_miembro].mean() if es_miembro.any() else np.nan
                    especificidad = (~dentro_c[~es_miembro]).mean() if (~es_miembro).any() else np.nan
                    filas_simca.append({
                        "Class": c,
                        "Components": modelo_c["n_comp"],
                        "Variance explained": f"{modelo_c['varianza_explicada_pct']:.1f}%",
                        "Calibration samples": modelo_c["n_muestras_calibracion"],
                        "Sensitivity (true members accepted)": round(sensibilidad, 3) if pd.notna(sensibilidad) else "n/a",
                        "Specificity (non-members rejected)": round(especificidad, 3) if pd.notna(especificidad) else "n/a",
                    })
                df_simca_resumen = pd.DataFrame(filas_simca)
                st.dataframe(df_simca_resumen, width='stretch')
                st.download_button(
                    "⬇️ Download SIMCA summary (Excel)",
                    data=df_a_excel_bytes({"simca_summary": df_simca_resumen}),
                    file_name="simca_summary.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="descargar_simca_excel",
                )

                st.markdown("**Sample assignment overview**")
                n_ambiguo = int((n_clases_aceptado > 1).sum())
                n_afuera = int((n_clases_aceptado == 0).sum())
                c1, c2, c3 = st.columns(3)
                c1.metric("Accepted by exactly 1 class", int((n_clases_aceptado == 1).sum()))
                c2.metric("Accepted by 2+ classes (ambiguous)", n_ambiguo)
                c3.metric("Rejected by all classes (outlier)", n_afuera)

                clase_detalle_simca = st.selectbox(
                    "View T² / Q plot for class", list(modelos_simca.keys()),
                    help="Shows the influence plot for that class's model — samples inside the "
                         "dashed box are accepted as members of this class.",
                )
                if clase_detalle_simca:
                    modelo_c = modelos_simca[clase_detalle_simca]
                    res_c = resultados_por_clase[clase_detalle_simca]
                    T2_lo, T2_hi = cu.rango_con_margen(res_c["T2"], modelo_c["T2_lim"])
                    Q_lo, Q_hi = cu.rango_con_margen(res_c["Q"], modelo_c["Q_lim"])
                    fig_simca = px.scatter(
                        x=res_c["T2"], y=res_c["Q"], hover_name=st.session_state["simca_ids_eval"],
                        color=como_texto(clases_eval_true), color_discrete_sequence=CLASS_PALETTE,
                        labels={"x": "T² (distance within class model)", "y": "Q (distance to class model)",
                                "color": "True class"},
                    )
                    fig_simca.add_vline(x=modelo_c["T2_lim"], line_dash="dash", line_color="red")
                    fig_simca.add_hline(y=modelo_c["Q_lim"], line_dash="dash", line_color="red")
                    fig_simca.update_xaxes(range=[T2_lo, T2_hi])
                    fig_simca.update_yaxes(range=[Q_lo, Q_hi])
                    fig_simca.update_layout(height=480, title=f"SIMCA model — class '{clase_detalle_simca}' "
                                                               "(samples inside the dashed box are accepted)")
                    st.plotly_chart(fig_simca, width='stretch')

                st.divider()
                st.markdown("**📄 Report for this SIMCA analysis**")
                if st.button("🖨️ Generate report (PDF)", key="generar_reporte_simca"):
                    secciones_simca = [
                        {"tipo": "titulo", "texto": "1. Dataset"},
                        {"tipo": "clave_valor", "pares": [
                            ("Samples used", st.session_state["simca_n_muestras"]),
                            ("Classes modeled", ", ".join(modelos_simca.keys())),
                            ("Preprocessing", " -> ".join(
                                p[0] for p in st.session_state["simca_pasos_pretratamiento"]) or "none"),
                            ("Target variance per class", f"{st.session_state['simca_varianza_objetivo']*100:.0f}%"),
                            ("Significance level (alpha)", st.session_state["simca_alpha"]),
                            ("Test proportion", f"{int(st.session_state['simca_prop_test']*100)}% "
                             f"({st.session_state['simca_metodo_split']})"),
                        ]},
                        {"tipo": "titulo", "texto": "2. Per-class model summary"},
                        {"tipo": "tabla", "encabezados": list(df_simca_resumen.columns),
                         "filas": df_simca_resumen.astype(str).values.tolist()},
                        {"tipo": "clave_valor", "pares": [
                            ("Accepted by exactly 1 class", int((n_clases_aceptado == 1).sum())),
                            ("Accepted by 2+ classes (ambiguous)", n_ambiguo),
                            ("Rejected by all classes (outlier)", n_afuera),
                        ]},
                    ]
                    for c, modelo_c in modelos_simca.items():
                        res_c = resultados_por_clase[c]
                        T2_lo, T2_hi = cu.rango_con_margen(res_c["T2"], modelo_c["T2_lim"])
                        Q_lo, Q_hi = cu.rango_con_margen(res_c["Q"], modelo_c["Q_lim"])
                        secciones_simca.append({"tipo": "salto_pagina"})
                        secciones_simca.append({"tipo": "subtitulo", "texto": f"Class '{c}'"})
                        secciones_simca.append({"tipo": "imagen", "fig": ru.fig_outliers(
                            res_c["T2"], res_c["Q"], modelo_c["T2_lim"], modelo_c["Q_lim"],
                            T2_lo, T2_hi, Q_lo, Q_hi, clases=como_texto(clases_eval_true))})
                    pdf_simca = ru.generar_reporte(
                        "SIMCA Report", f"Classes: {', '.join(modelos_simca.keys())}", secciones_simca,
                    )
                    st.download_button(
                        "⬇️ Download full report (PDF)",
                        data=bytes(pdf_simca.output()),
                        file_name="simca_report.pdf", mime="application/pdf",
                        key="descargar_reporte_simca",
                    )
                st.divider()

                nombre_guardado_simca = st.text_input("Name to save this SIMCA model set as", value="SIMCA")
                if st.button("💾 Save these SIMCA models for the Prediction tab"):
                    id_trazabilidad_simca = mu.generar_id_trazabilidad()
                    ficha_simca = {
                        "id_trazabilidad": id_trazabilidad_simca,
                        "fecha_creacion": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "nombre_modelo": nombre_guardado_simca,
                        "tipo": "simca",
                        "algoritmo": "SIMCA",
                        "n_muestras": st.session_state["simca_n_muestras"],
                        "n_variables_totales": int(X_modelado_simca.shape[1]),
                        "n_variables_usadas": int(X_modelado_simca.shape[1]),
                        "descripcion_y": ", ".join(f"{c}: {int(n)}" for c, n in conteos_simca.items()),
                        "pretratamiento_desc": " -> ".join(
                            p[0] for p in st.session_state["simca_pasos_pretratamiento"]) or "none",
                        "seleccion_variables_desc": "none (SIMCA uses full spectrum per class)",
                        "hiperparametros": {"target_variance": st.session_state["simca_varianza_objetivo"],
                                            "alpha": st.session_state["simca_alpha"]},
                        "cv_desc": "n/a (class-modeling, not cross-validated the same way)",
                        "prop_test_desc": f"{int(st.session_state['simca_prop_test']*100)}% "
                                          f"({st.session_state['simca_metodo_split']})",
                        "metricas": {f"Sensitivity ({row['Class']})": row["Sensitivity (true members accepted)"]
                                     for row in filas_simca}
                                    | {f"Specificity ({row['Class']})": row["Specificity (non-members rejected)"]
                                       for row in filas_simca},
                        "entorno_software": mu.info_entorno_software(),
                    }
                    st.session_state.modelos_guardados[nombre_guardado_simca] = {
                        "tipo": "simca",
                        "modelo": modelos_simca,
                        "mascara_variables": None,
                        "numeros_onda": st.session_state["simca_eje_usado"],
                        "pasos_pretratamiento": st.session_state["simca_pasos_pretratamiento"],
                        "ficha": ficha_simca,
                    }
                    st.success(f"SIMCA model set '{nombre_guardado_simca}' saved (traceability ID: "
                               f"{id_trazabilidad_simca}). It is now available in the Prediction tab.")

# TAB: REGRESSION
# -----------------------------------------------------------------------
with tabs[9]:
    st.subheader("Supervised regression")
    if st.button("🔄 Reset this tab", key="reset_reg",
                 help="Clears all trained models, metrics, and plots from this tab, so you can "
                      "start a completely fresh run without any leftover results from before."):
        resetear_prefijo("reg_")
        st.rerun()

    if st.session_state.valores_y is None:
        st.markdown("**Reference value (continuous Y variable)**")
        st.caption("Not loaded yet. Easiest: go back to the sidebar's 'Reference value column' "
                   "selector when (re-)loading your data file. Or define it here instead:")
        modo_y = st.radio("How do you want to load the reference values?",
                           ["Enter manually", "Upload file (id, value)"], horizontal=True,
                           help="The Y variable you want to predict (e.g. a lab-measured concentration) "
                                "for each sample, in the same order as your spectra.")

        if modo_y == "Enter manually":
            st.caption("One value per sample, comma-separated, in the same order as the IDs:")
            st.code(", ".join(st.session_state.ids[:8]) + (", ..." if len(st.session_state.ids) > 8 else ""))
            texto_y = st.text_area("Values (comma-separated)", key="texto_valores_y")
            if texto_y.strip():
                try:
                    lista_y = [float(v.strip()) for v in texto_y.split(",")]
                    if len(lista_y) != len(st.session_state.ids):
                        st.error(f"You entered {len(lista_y)} values but there are {len(st.session_state.ids)} samples.")
                    else:
                        st.session_state.valores_y = np.array(lista_y, dtype=float)
                        st.rerun()
                except ValueError:
                    st.error("Some value could not be parsed as a number.")
        else:
            archivo_y = st.file_uploader(
                "File with columns: id, value", type=["csv", "xlsx", "xls"], key="valores_y_csv",
            )
            if archivo_y is not None:
                try:
                    if archivo_y.name.lower().endswith(".csv"):
                        df_y = pd.read_csv(archivo_y)
                    else:
                        df_y = pd.read_excel(archivo_y)
                    df_y = df_y.set_index(df_y.columns[0])
                    df_y.index = df_y.index.astype(str)
                    mapa_y = df_y.iloc[:, 0].to_dict()
                    valores_y = np.array([mapa_y.get(str(i), np.nan) for i in st.session_state.ids], dtype=float)
                    n_faltantes = int(np.isnan(valores_y).sum())
                    if n_faltantes > 0:
                        st.warning(f"{n_faltantes} sample(s) without a reference value in the file — "
                                   "they are automatically excluded from the regression.")
                    st.session_state.valores_y = valores_y
                    st.rerun()
                except Exception as e:
                    st.error(f"Error reading the file: {e}")
    else:
        col_y1, col_y2 = st.columns([4, 1])
        col_y1.success(f"✓ Reference values loaded for {len(st.session_state.valores_y)} samples.")
        if col_y2.button("Change", help="Clear the loaded reference values to enter or upload different ones."):
            st.session_state.valores_y = None
            st.rerun()

    if st.session_state.valores_y is None:
        st.info("Load the reference values to be able to train a regression model.")
    else:
        ids_activos, _, _ = datos_activos()
        X_modelado = st.session_state.X_pret[indice_activo()]
        eje_modelado = st.session_state.numeros_onda_pret
        y_activos = st.session_state.valores_y[indice_activo()]

        mask_validos = ~np.isnan(y_activos)
        if mask_validos.sum() < 5:
            st.error("Too few samples with a valid reference value (at least 5 are needed).")
        else:
            ids_reg = ids_activos[mask_validos]
            X_reg = X_modelado[mask_validos]
            y_reg = y_activos[mask_validos]
            st.caption(f"Using {mask_validos.sum()} of {len(y_activos)} active samples (with a reference value).")

            col1, col2 = st.columns(2)
            with col1:
                modelos_elegidos_r = st.multiselect(
                    "Models to compare",
                    list(mu.crear_regresores().keys()),
                    default=["PLS", "Ridge", "Random Forest"],
                    help="Pick one or more algorithms to train and compare side by side in the same "
                         "run — the results table below will show every metric for each of them.",
                )
                metodo_seleccion_r = st.selectbox(
                    "Variable selection", ["None", "Boruta", "Genetic Algorithm"], key="sel_var_reg",
                    help="Reduce the spectrum to just the most informative wavenumbers before "
                         "training. Boruta is conservative and keeps anything that helps at all; "
                         "the Genetic Algorithm explores combinations more freely but is slower and "
                         "a bit more random. 'None' uses the full spectrum.",
                )
            with col2:
                cv_folds_sel_r = st.selectbox(
                    "Cross-validation folds (internal)",
                    [3, 5, 10, "LOO (leave 1 sample out at a time)"], index=1, key="cv_reg",
                    help="How many folds to use for cross-validation. More folds = less biased but "
                         "slower and more variance between folds. LOO (Leave-One-Out) trains one "
                         "model per sample — most thorough, slowest, and typical for small datasets.",
                )
                cv_folds_r = "LOO" if cv_folds_sel_r == "LOO (leave 1 sample out at a time)" else cv_folds_sel_r
                if cv_folds_r == "LOO" and X_reg.shape[0] > 150:
                    st.caption("⚠ LOO trains one model per sample — with many samples this can take a while.")
                prop_test_r = st.slider("Proportion for independent test set (0 = use everything for CV)",
                                         0.0, 0.4, 0.2, step=0.05, key="test_reg",
                                         help="Fraction of samples set aside BEFORE training, never used "
                                              "to choose anything — only to get one final, honest check "
                                              "of how the model performs on unseen data. 0 means skip "
                                              "this and use every sample for cross-validation instead.")
                metodo_split_sel_r = st.selectbox(
                    "Test set selection", ["Random", "SPXY (extended Kennard-Stone)"],
                    disabled=(prop_test_r == 0), key="split_reg",
                    help="How to choose which samples go into the test set. 'Random' picks them by "
                         "chance. 'SPXY' is an extended version of Kennard-Stone made specifically "
                         "for regression (Galvão et al., 2005): plain Kennard-Stone only looks at "
                         "the spectra, with no guarantee the reference value (Y) range is well "
                         "represented too; SPXY adds the Y range into the same calculation, so the "
                         "training set is spread out across BOTH the spectral space AND the Y range "
                         "at once. It's the standard, widely-used way to fix that gap.",
                )
                metodo_split_r = "kennard_stone" if metodo_split_sel_r.startswith("SPXY") else "random"
                if metodo_split_r == "kennard_stone":
                    st.caption("ℹ️ SPXY (an extended Kennard-Stone for regression) picks the TRAINING "
                               "set to broadly cover BOTH the spectral space and the reference value (Y) "
                               "range at once (it will include the more 'extreme' samples in either "
                               "sense); the remaining, more 'typical' samples become the test set.")
                optimizar_r = st.checkbox(
                    "Optimize hyperparameters (slower)", key="opt_reg",
                    help="Automatically search for better settings for each model instead of using "
                         "the defaults. Usually improves results a bit, at the cost of extra "
                         "computation time.",
                )
                preset_opt_r = st.selectbox(
                    "Optimization budget", list(PRESETS_OPTIMIZACION.keys()), index=1,
                    disabled=not optimizar_r, key="metodo_opt_reg",
                    help="How much effort to spend searching for good hyperparameters — see the "
                         "explanation below once you pick one.",
                )
                metodo_opt_r = PRESETS_OPTIMIZACION[preset_opt_r][0]

            if optimizar_r:
                st.caption(f"ℹ️ **{preset_opt_r}**: {PRESETS_OPTIMIZACION[preset_opt_r][1]}")

            if optimizar_r and any(m in mu.ALGORITMOS_LENTOS_AL_OPTIMIZAR for m in modelos_elegidos_r):
                lentos_elegidos_r = [m for m in modelos_elegidos_r if m in mu.ALGORITMOS_LENTOS_AL_OPTIMIZAR]
                extra_r = " With LOO this can take several minutes." if cv_folds_r == "LOO" else ""
                st.caption(f"⚠ Optimizing hyperparameters for {', '.join(lentos_elegidos_r)} can still take a "
                           f"while on a shared/limited CPU. If it's too slow, try 'Fast', or fewer folds.{extra_r}")

            if optimizar_r and len(modelos_elegidos_r) > 3:
                st.warning(f"⚠️ You're optimizing hyperparameters for {len(modelos_elegidos_r)} models at once. "
                           "Each one runs its own search in parallel across CPU cores, so doing several "
                           "together can exhaust the machine's memory and cause the app to crash or freeze. "
                           "We recommend optimizing **up to 3 models at a time** — train the rest with "
                           "default settings first, then optimize the best candidates separately.")

            if metodo_seleccion_r == "Boruta":
                c1, c2 = st.columns(2)
                boruta_max_iter_r = c1.slider(
                    "Boruta iterations", 10, 500, 40, step=10, key="boruta_iter_reg",
                    help="How many comparison rounds to run against the random 'shadow' variables. "
                         "More iterations give a more stable decision but take longer.",
                )
                boruta_alpha_r = c2.select_slider(
                    "Boruta p-value (alpha)", options=[0.01, 0.02, 0.05, 0.1], value=0.05, key="boruta_alpha_reg",
                    help="Significance level for keeping a variable: smaller (e.g. 0.01) is stricter "
                         "and keeps fewer, more confidently relevant variables; larger (e.g. 0.1) is "
                         "more permissive.",
                )
            if metodo_seleccion_r == "Genetic Algorithm":
                c1, c2 = st.columns(2)
                ga_poblacion_r = c1.slider("Population size", 10, 60, 20, key="ga_pob_reg")
                ga_generaciones_r = c2.slider("Generations", 5, 50, 15, key="ga_gen_reg")

            if st.session_state.get("reg_resultados") is not None:
                insignia_estado("reg_firma", "This trained regression model")

            if st.button("🚀 Train and evaluate (Regression)", disabled=len(modelos_elegidos_r) == 0):
                for _clave in ["reg_pvalores", "reg_puntajes_cv", "reg_comparacion_metodo",
                                "reg_curva_aprendizaje", "reg_curva_modelo", "reg_ultima_ficha"]:
                    st.session_state.pop(_clave, None)
                mascara_variables_r = None
                _msg_espera_r = "This may take a few minutes..." if (optimizar_r or metodo_seleccion_r in ("Boruta", "Genetic Algorithm")) else "Training..."
                with st.spinner(f"Selecting variables ({_msg_espera_r})" if metodo_seleccion_r != "None" else _msg_espera_r):
                    if metodo_seleccion_r == "Boruta":
                        try:
                            mascara_variables_r = mu.seleccionar_variables_boruta(
                                X_reg, y_reg, es_clasificacion=False,
                                max_iter=boruta_max_iter_r, alpha=boruta_alpha_r,
                            )
                            if mascara_variables_r.sum() == 0:
                                st.warning("Boruta did not select any variable; using all of them.")
                                mascara_variables_r = None
                        except Exception as e:
                            st.error(f"Boruta failed ({e}); using all variables.")
                    elif metodo_seleccion_r == "Genetic Algorithm":
                        from sklearn.linear_model import LinearRegression as _LR
                        cv_ga_r = 3 if cv_folds_r == "LOO" else min(3, cv_folds_r)
                        ga_r = mu.SeleccionGenetica(
                            _LR(), es_clasificacion=False,
                            tam_poblacion=ga_poblacion_r, n_generaciones=ga_generaciones_r,
                            cv=cv_ga_r, random_state=0,
                        )
                        ga_r.fit(X_reg, y_reg)
                        mascara_variables_r = ga_r.mejor_mascara_

                    X_sel_r = X_reg[:, mascara_variables_r] if mascara_variables_r is not None else X_reg

                    resultados_r = {}
                    hiperparametros_optimos_r = {}
                    descripcion_opt_usada_r = {}
                    catalogo_r = mu.crear_regresores()
                    for nombre in modelos_elegidos_r:
                        modelo = catalogo_r[nombre]
                        try:
                            if optimizar_r and nombre in mu.GRILLAS_REGRESION:
                                modelo, mejores_params_r, _, desc_opt_r = mu.optimizar_hiperparametros(
                                    modelo, mu.GRILLAS_REGRESION[nombre], X_sel_r, y_reg,
                                    es_clasificacion=False, cv=cv_folds_r, metodo=metodo_opt_r,
                                )
                                hiperparametros_optimos_r[nombre] = mejores_params_r
                                descripcion_opt_usada_r[nombre] = desc_opt_r
                            resultados_r[nombre] = mu.entrenar_evaluar_regresion(
                                modelo, X_sel_r, y_reg, ids=ids_reg,
                                cv=cv_folds_r, proporcion_test=prop_test_r, metodo_split=metodo_split_r,
                            )
                        except Exception as e:
                            resultados_r[nombre] = {"error": str(e)}

                st.session_state["reg_resultados"] = resultados_r
                st.session_state["reg_mascara_variables"] = mascara_variables_r
                st.session_state["reg_descripcion_opt"] = descripcion_opt_usada_r
                st.session_state["reg_eje_usado"] = eje_modelado
                st.session_state["reg_pasos_pretratamiento"] = st.session_state.pasos_pretratamiento
                st.session_state["reg_hiperparametros"] = hiperparametros_optimos_r
                st.session_state["reg_espectro_promedio"] = X_reg.mean(axis=0)
                st.session_state["reg_cv_folds"] = cv_folds_r
                st.session_state["reg_prop_test"] = prop_test_r
                st.session_state["reg_metodo_split"] = metodo_split_sel_r
                st.session_state["reg_metodo_seleccion"] = metodo_seleccion_r
                st.session_state["reg_n_muestras"] = X_reg.shape[0]
                st.session_state["reg_firma"] = firma_datos_activos()

            if "reg_resultados" in st.session_state:
                resultados_r = st.session_state["reg_resultados"]
                mascara_variables_r = st.session_state["reg_mascara_variables"]
                hiperparametros_optimos_r = st.session_state.get("reg_hiperparametros", {})
                descripcion_opt_usada_r = st.session_state.get("reg_descripcion_opt", {})

                if mascara_variables_r is not None:
                    st.caption(f"Selected variables: {int(mascara_variables_r.sum())} of {len(mascara_variables_r)}.")
                    with st.expander("📍 View selected variables on the mean spectrum"):
                        fig_vars_r = go.Figure()
                        eje_rr = st.session_state["reg_eje_usado"]
                        fig_vars_r.add_trace(go.Scatter(x=eje_rr, y=st.session_state["reg_espectro_promedio"],
                                                         mode="lines", line=dict(color="#5F5E5A"), name="Mean spectrum"))
                        idx_sel_r = np.where(mascara_variables_r)[0]
                        bloques_r = np.split(idx_sel_r, np.where(np.diff(idx_sel_r) != 1)[0] + 1) if len(idx_sel_r) else []
                        for j, bloque in enumerate(bloques_r):
                            x0, x1 = eje_rr[bloque[0]], eje_rr[bloque[-1]]
                            fig_vars_r.add_vrect(x0=min(x0, x1), x1=max(x0, x1), fillcolor="#0F6E56",
                                                  opacity=0.25, line_width=0)
                        fig_vars_r.update_layout(height=380, xaxis_title="Wavenumber", yaxis_title="Signal")
                        if eje_rr[0] > eje_rr[-1]:
                            fig_vars_r.update_xaxes(autorange="reversed")
                        st.plotly_chart(fig_vars_r, width='stretch')

                        st.markdown("**List of selected variables**")
                        df_vars_sel_r = df_variables_seleccionadas(eje_rr, mascara_variables_r)
                        st.dataframe(df_vars_sel_r, width='stretch', height=200)
                        st.download_button(
                            "⬇️ Download selected variables (Excel)",
                            data=df_a_excel_bytes({"selected_variables": df_vars_sel_r}),
                            file_name="selected_variables_regression.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="descargar_vars_reg",
                        )

                filas = []
                for nombre, res in resultados_r.items():
                    if "error" in res:
                        filas.append({"Model": nombre, "Error": res["error"]})
                        continue
                    fila = {
                        "Model": nombre,
                        "RMSE (CV)": round(res["cv"]["rmse_cv"], 4),
                        "R² (CV)": round(res["cv"]["r2_cv"], 3),
                        "RPD (CV)": round(res["cv"]["rpd_cv"], 2),
                        "RPIQ (CV)": round(res["cv"]["rpiq_cv"], 2),
                    }
                    if "test" in res:
                        fila["RMSE (test)"] = round(res["test"]["rmse"], 4)
                        fila["R² (test)"] = round(res["test"]["r2"], 3)
                        fila["RPD (test)"] = round(res["test"]["rpd"], 2)
                    filas.append(fila)
                df_metricas_reg = pd.DataFrame(filas)
                st.dataframe(df_metricas_reg, width='stretch')
                st.download_button(
                    "⬇️ Download metrics table (Excel)",
                    data=df_a_excel_bytes({"regression_metrics": df_metricas_reg}),
                    file_name="regression_metrics.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="descargar_metricas_reg",
                )

                modelos_exitosos_r = [n for n in resultados_r if "error" not in resultados_r[n]]
                if len(modelos_exitosos_r) >= 2:
                    hay_test_reg = "test" in resultados_r[modelos_exitosos_r[0]]
                    with st.expander("📊 Compare models statistically"):
                        if hay_test_reg:
                            st.caption("Compares models on the held-out TEST set predictions, sample by "
                                       "sample (paired t-test on squared errors) — this is what can "
                                       "actually reveal overfitting, since two models can look equally "
                                       "good in cross-validation yet behave very differently on genuinely "
                                       "unseen data. A p-value below 0.05 means the difference is unlikely "
                                       "to be just chance.")
                        else:
                            st.caption("No independent test set was configured, so this compares models "
                                       "using a paired t-test on their per-fold cross-validation scores "
                                       "instead (same folds for every model). Note this can't detect "
                                       "overfitting the way a held-out test set would — consider setting "
                                       "aside a test proportion above 0% for a more trustworthy comparison.")
                        if st.button("Run comparison", key="comparar_modelos_reg"):
                            if hay_test_reg:
                                pvalores_r, puntajes_r = mu.comparar_modelos_en_test(
                                    {n: resultados_r[n] for n in modelos_exitosos_r}, es_clasificacion=False,
                                )
                            else:
                                mascara_cmp_r = st.session_state["reg_mascara_variables"]
                                ids_train_ref_r = resultados_r[modelos_exitosos_r[0]]["cv"]["ids"]
                                mask_train_cmp_r = np.isin(ids_reg, ids_train_ref_r)
                                X_cmp_r = X_reg[mask_train_cmp_r]
                                y_cmp_r = y_reg[mask_train_cmp_r]
                                if mascara_cmp_r is not None:
                                    X_cmp_r = X_cmp_r[:, mascara_cmp_r]
                                modelos_para_comparar_r = {
                                    n: mu.clone(resultados_r[n]["modelo_final"]) for n in modelos_exitosos_r
                                }
                                with st.spinner("Running cross-validation for every model on the same folds..."):
                                    pvalores_r, puntajes_r = mu.comparar_modelos_estadisticamente(
                                        modelos_para_comparar_r, X_cmp_r, y_cmp_r, es_clasificacion=False,
                                        cv=st.session_state["reg_cv_folds"],
                                    )
                            st.session_state["reg_pvalores"] = pvalores_r
                            st.session_state["reg_puntajes_cv"] = puntajes_r
                            st.session_state["reg_comparacion_metodo"] = (
                                "Paired t-test on squared errors on the held-out test set" if hay_test_reg
                                else "Paired t-test on cross-validation folds (no test set configured)"
                            )
                        if "reg_pvalores" in st.session_state:
                            interpretacion_r = mu.interpretar_comparacion_modelos(
                                st.session_state["reg_pvalores"], st.session_state["reg_puntajes_cv"],
                            )
                            if interpretacion_r["hay_diferencias"]:
                                for msg in interpretacion_r["mensajes"]:
                                    st.success(msg)
                            else:
                                st.info(interpretacion_r["mensajes"][0])
                            st.markdown("**p-values:**")
                            st.dataframe(st.session_state["reg_pvalores"].style.format("{:.4f}"), width='stretch')

                nombre_detalle_r = st.selectbox(
                    "View details for", [n for n in resultados_r if "error" not in resultados_r[n]], key="detalle_reg",
                    help="Choose which of the trained models to inspect below: predicted vs. actual, "
                         "hyperparameters, learning curve, Williams plot, and the report/save options.",
                )
                if nombre_detalle_r:
                    res = resultados_r[nombre_detalle_r]

                    if hiperparametros_optimos_r.get(nombre_detalle_r):
                        with st.container(border=True):
                            st.markdown(f"**⚙️ Optimized hyperparameters — {nombre_detalle_r}**")
                            cols_hp_r = st.columns(len(hiperparametros_optimos_r[nombre_detalle_r]))
                            for col_hp, (k, v) in zip(cols_hp_r, hiperparametros_optimos_r[nombre_detalle_r].items()):
                                col_hp.metric(k, str(v))
                            st.caption(f"🔧 Method: {descripcion_opt_usada_r.get(nombre_detalle_r, 'n/a')}")

                    hay_test_r = "test" in res
                    y_t_cv, y_p_cv, ids_cv = res["cv"]["y_true"], res["cv"]["y_pred"], res["cv"]["ids"]

                    fig_pred = go.Figure()
                    fig_pred.add_trace(go.Scatter(
                        x=y_t_cv, y=y_p_cv, mode="markers", name="Training (CV)",
                        marker=dict(color="#185FA5", size=8),
                        text=ids_cv, hovertemplate="%{text}<br>Actual: %{x}<br>Predicted: %{y}<extra></extra>",
                    ))
                    todos_valores = [y_t_cv, y_p_cv]
                    if hay_test_r:
                        y_t_test, y_p_test, ids_test = res["test"]["y_true"], res["test"]["y_pred"], res["test"]["ids"]
                        fig_pred.add_trace(go.Scatter(
                            x=y_t_test, y=y_p_test, mode="markers", name="Test",
                            marker=dict(color="#D97706", size=9, symbol="diamond"),
                            text=ids_test, hovertemplate="%{text}<br>Actual: %{x}<br>Predicted: %{y}<extra></extra>",
                        ))
                        todos_valores += [y_t_test, y_p_test]

                    lim_lo = min(arr.min() for arr in todos_valores)
                    lim_hi = max(arr.max() for arr in todos_valores)
                    fig_pred.add_trace(go.Scatter(x=[lim_lo, lim_hi], y=[lim_lo, lim_hi], mode="lines",
                                                   line=dict(dash="dash", color="gray"), name="Ideal (y=x)"))
                    fig_pred.update_layout(height=470, title=f"Predicted vs. actual — {nombre_detalle_r}",
                                            xaxis_title="Actual value", yaxis_title="Predicted value")
                    st.plotly_chart(fig_pred, width='stretch')

                    df_exp_cv = mu.exportar_predicciones_regresion(ids_cv, y_t_cv, y_p_cv)
                    if hay_test_r:
                        col_dl_cv, col_dl_test = st.columns(2)
                    else:
                        col_dl_cv = st.container()
                    col_dl_cv.download_button(
                        f"⬇️ Download predictions ({nombre_detalle_r}, Training/CV)",
                        data=df_exp_cv.to_csv(index=False).encode("utf-8"),
                        file_name=f"predictions_{nombre_detalle_r}_CV.csv", mime="text/csv",
                        key="descargar_pred_reg_cv",
                    )
                    if hay_test_r:
                        df_exp_test = mu.exportar_predicciones_regresion(ids_test, y_t_test, y_p_test)
                        col_dl_test.download_button(
                            f"⬇️ Download predictions ({nombre_detalle_r}, Test)",
                            data=df_exp_test.to_csv(index=False).encode("utf-8"),
                            file_name=f"predictions_{nombre_detalle_r}_Test.csv", mime="text/csv",
                            key="descargar_pred_reg_test",
                        )

                    # Diagnostics further below (Williams plot, PDF report, model card) need a
                    # single dataset — prefer Test when available (a more honest check of
                    # generalization), otherwise fall back to the CV predictions.
                    if hay_test_r:
                        y_t, y_p, ids_f, fuente_r = y_t_test, y_p_test, ids_test, "Test"
                    else:
                        y_t, y_p, ids_f, fuente_r = y_t_cv, y_p_cv, ids_cv, "Training (CV)"

                    with st.expander(f"📈 Learning curve — {nombre_detalle_r}"):
                        st.caption("Shows performance vs. how many training samples were used. If both "
                                   "curves are still rising and far apart, more samples would likely help. "
                                   "If they're flat and close together, the model has plateaued.")
                        if st.button("Compute learning curve", key="curva_aprendizaje_reg"):
                            mascara_lc_r = st.session_state["reg_mascara_variables"]
                            X_lc_r = X_reg[:, mascara_lc_r] if mascara_lc_r is not None else X_reg
                            with st.spinner("Training with increasing sample sizes (this may take a few minutes)..."):
                                curva_r = mu.calcular_curva_aprendizaje(
                                    mu.clone(res["modelo_final"]), X_lc_r, y_reg, es_clasificacion=False,
                                    cv=min(5, st.session_state["reg_cv_folds"]) if st.session_state["reg_cv_folds"] != "LOO" else 5,
                                )
                            st.session_state["reg_curva_aprendizaje"] = curva_r
                            st.session_state["reg_curva_modelo"] = nombre_detalle_r
                        if st.session_state.get("reg_curva_modelo") == nombre_detalle_r and "reg_curva_aprendizaje" in st.session_state:
                            curva_r = st.session_state["reg_curva_aprendizaje"]
                            fig_lc_r = go.Figure()
                            fig_lc_r.add_trace(go.Scatter(x=curva_r["train_sizes"], y=curva_r["train_scores_mean"],
                                                           mode="lines+markers", name="Training score",
                                                           line=dict(color="#185FA5")))
                            fig_lc_r.add_trace(go.Scatter(x=curva_r["train_sizes"], y=curva_r["val_scores_mean"],
                                                           mode="lines+markers", name="Cross-validation score",
                                                           line=dict(color="#B91C1C")))
                            fig_lc_r.update_layout(height=380, xaxis_title="Training set size (samples)",
                                                    yaxis_title=curva_r["scoring"])
                            st.plotly_chart(fig_lc_r, width='stretch')

                    with st.expander(f"📐 Williams plot (leverage vs. residual) — {nombre_detalle_r}"):
                        st.caption("A classic applicability-domain diagnostic (common in QSAR/chemometrics): "
                                   "leverage (X-axis) measures how far a sample sits from the center of the "
                                   "spectral space — high leverage means an unusual spectrum. The standardized "
                                   "residual (Y-axis) measures how badly the model predicted that sample. "
                                   "Samples with HIGH leverage AND a large residual ('bad leverage points', top "
                                   "or bottom right) are the ones most likely to be distorting the model — "
                                   "worth double-checking. High leverage with a small residual ('good leverage "
                                   "points') are fine; they just extend the calibration range.")
                        indices_wp = [np.where(ids_reg == i)[0][0] for i in ids_f]
                        X_wp = X_reg[indices_wp]
                        mascara_wp_r = st.session_state["reg_mascara_variables"]
                        if mascara_wp_r is not None:
                            X_wp = X_wp[:, mascara_wp_r]
                        leverage_wp, n_comp_wp = cu.calcular_leverage(X_wp)
                        residuo_wp = cu.calcular_residuo_estandarizado(y_t, y_p)
                        lim_leverage_wp = cu.limite_leverage(len(leverage_wp), n_comp_wp)

                        fig_wp = px.scatter(x=leverage_wp, y=residuo_wp, hover_name=ids_f,
                                             labels={"x": "Leverage", "y": "Standardized residual"})
                        fig_wp.add_vline(x=lim_leverage_wp, line_dash="dash", line_color="red")
                        fig_wp.add_hline(y=3, line_dash="dash", line_color="red")
                        fig_wp.add_hline(y=-3, line_dash="dash", line_color="red")
                        fig_wp.update_layout(height=450, title=f"Williams plot — {nombre_detalle_r} ({fuente_r})")
                        st.plotly_chart(fig_wp, width='stretch')

                    st.divider()
                    st.markdown("**📄 Report for this analysis**")
                    st.caption("Includes: dataset used, spectra, preprocessing, selected variables, "
                               "a comparison table of all trained models, and the detail "
                               f"({nombre_detalle_r}) with predicted vs. actual and hyperparameters.")
                    if st.button("🖨️ Generate report (PDF)", key="generar_reporte_reg"):
                        secciones_rep_r = [
                            {"tipo": "titulo", "texto": "1. Dataset"},
                            {"tipo": "clave_valor", "pares": [
                                ("Samples used", int(X_reg.shape[0])),
                                ("Original variables", int(X_reg.shape[1])),
                                ("Y variable", f"n={len(y_reg)}, mean={y_reg.mean():.4g}, "
                                               f"std={y_reg.std():.4g}, "
                                               f"range=[{y_reg.min():.4g}, {y_reg.max():.4g}]"),
                                ("Preprocessing", " -> ".join(
                                    p[0] for p in st.session_state["reg_pasos_pretratamiento"]) or "none"),
                            ]},
                            {"tipo": "imagen", "fig": ru.fig_espectros(eje_modelado, X_reg, titulo="Preprocessed spectra")},
                            {"tipo": "titulo", "texto": "2. Variable selection"},
                            {"tipo": "clave_valor", "pares": [
                                ("Method", st.session_state["reg_metodo_seleccion"]),
                                ("Selected variables", int(mascara_variables_r.sum()) if mascara_variables_r is not None
                                 else f"all ({X_reg.shape[1]})"),
                            ]},
                        ]
                        if mascara_variables_r is not None:
                            secciones_rep_r.append({"tipo": "imagen", "fig": ru.fig_variables_seleccionadas(
                                eje_modelado, X_reg.mean(axis=0), mascara_variables_r)})
                            idx_sel_rep_r = np.where(mascara_variables_r)[0]
                            secciones_rep_r.append({"tipo": "tabla",
                                "encabezados": ["Position", "Wavenumber"],
                                "filas": [[str(i), f"{eje_modelado[i]:.2f}"] for i in idx_sel_rep_r[:60]],
                                "anchos": [30, 70]})
                            if len(idx_sel_rep_r) > 60:
                                secciones_rep_r.append({"tipo": "parrafo",
                                    "texto": f"(showing the first 60 of {len(idx_sel_rep_r)} variables — "
                                             "the full list is in the Excel file downloadable from the app)"})
                        secciones_rep_r += [
                            {"tipo": "salto_pagina"},
                            {"tipo": "titulo", "texto": "3. Model comparison"},
                            {"tipo": "clave_valor", "pares": [
                                ("Cross-validation", f"{res['cv']['cv_folds_usados']}-fold" if cv_folds_r != "LOO"
                                 else f"Leave-One-Out ({res['cv']['cv_folds_usados']} repetitions)"),
                                ("Test proportion", f"{int(prop_test_r * 100)}% ({metodo_split_sel_r})"),
                            ]},
                            {"tipo": "tabla",
                             "encabezados": list(df_metricas_reg.columns),
                             "filas": df_metricas_reg.astype(str).values.tolist()},
                            {"tipo": "titulo", "texto": f"4. Detail: {nombre_detalle_r}"},
                            {"tipo": "imagen", "fig": ru.fig_predicho_vs_real(y_t, y_p)},
                        ]
                        if hiperparametros_optimos_r.get(nombre_detalle_r):
                            secciones_rep_r.append({"tipo": "subtitulo", "texto": "Optimal hyperparameters"})
                            secciones_rep_r.append({"tipo": "clave_valor",
                                "pares": [("Optimization method", descripcion_opt_usada_r.get(nombre_detalle_r, "n/a"))]
                                + list(hiperparametros_optimos_r[nombre_detalle_r].items())})
                        if st.session_state.get("reg_curva_modelo") == nombre_detalle_r and "reg_curva_aprendizaje" in st.session_state:
                            secciones_rep_r.append({"tipo": "subtitulo", "texto": f"Learning curve — {nombre_detalle_r}"})
                            secciones_rep_r.append({"tipo": "imagen",
                                "fig": ru.fig_curva_aprendizaje(st.session_state["reg_curva_aprendizaje"], nombre_detalle_r)})
                        if "reg_pvalores" in st.session_state:
                            secciones_rep_r.append({"tipo": "salto_pagina"})
                            secciones_rep_r.append({"tipo": "titulo", "texto": "5. Statistical comparison between models"})
                            secciones_rep_r.append({"tipo": "parrafo",
                                "texto": f"Method: {st.session_state.get('reg_comparacion_metodo', 'n/a')}. "
                                         "A p-value below 0.05 suggests the performance difference between "
                                         "that pair of models is unlikely to be due to chance alone."})
                            interpretacion_rep_r = mu.interpretar_comparacion_modelos(
                                st.session_state["reg_pvalores"], st.session_state["reg_puntajes_cv"],
                            )
                            for msg in interpretacion_rep_r["mensajes"]:
                                secciones_rep_r.append({"tipo": "parrafo", "texto": msg.replace("**", "")})
                            secciones_rep_r.append({"tipo": "imagen",
                                "fig": ru.fig_comparacion_pvalores(st.session_state["reg_pvalores"])})

                        pdf_reporte_reg = ru.generar_reporte(
                            "Regression Report", f"Models: {', '.join(modelos_elegidos_r)}", secciones_rep_r,
                        )
                        st.download_button(
                            "⬇️ Download full report (PDF)",
                            data=bytes(pdf_reporte_reg.output()),
                            file_name="regression_report.pdf", mime="application/pdf",
                            key="descargar_reporte_reg",
                        )
                    st.divider()

                    nombre_guardado_r = st.text_input("Name to save this model as", value=nombre_detalle_r,
                                                       key="nombre_guardado_reg")
                    if st.button("💾 Save this model for the Prediction tab", key="guardar_reg"):
                        id_trazabilidad_r = mu.generar_id_trazabilidad()
                        y_ref_activos = y_reg
                        desc_y_r = (f"n={len(y_ref_activos)}, mean={y_ref_activos.mean():.4g}, "
                                    f"std={y_ref_activos.std():.4g}, "
                                    f"range=[{y_ref_activos.min():.4g}, {y_ref_activos.max():.4g}]")
                        cv_desc_r = (f"Leave-One-Out ({res['cv']['cv_folds_usados']} repetitions)"
                                     if cv_folds_r == "LOO" else f"{res['cv']['cv_folds_usados']}-fold")
                        metricas_ficha_r = {
                            "RMSE (CV)": round(res["cv"]["rmse_cv"], 4),
                            "R2 (CV)": round(res["cv"]["r2_cv"], 3),
                            "RPD (CV)": round(res["cv"]["rpd_cv"], 2),
                            "RPIQ (CV)": round(res["cv"]["rpiq_cv"], 2),
                        }
                        if "test" in res:
                            metricas_ficha_r.update({
                                "RMSE (test)": round(res["test"]["rmse"], 4),
                                "R2 (test)": round(res["test"]["r2"], 3),
                                "RPD (test)": round(res["test"]["rpd"], 2),
                            })
                        ficha_r = {
                            "id_trazabilidad": id_trazabilidad_r,
                            "fecha_creacion": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "nombre_modelo": nombre_guardado_r,
                            "tipo": "regression",
                            "algoritmo": nombre_detalle_r,
                            "n_muestras": int(X_reg.shape[0]),
                            "n_variables_totales": int(X_reg.shape[1]),
                            "n_variables_usadas": int(mascara_variables_r.sum()) if mascara_variables_r is not None else int(X_reg.shape[1]),
                            "descripcion_y": desc_y_r,
                            "pretratamiento_desc": " -> ".join(p[0] for p in st.session_state["reg_pasos_pretratamiento"]) or "none",
                            "seleccion_variables_desc": st.session_state["reg_metodo_seleccion"],
                            "hiperparametros": hiperparametros_optimos_r.get(nombre_detalle_r, {}),
                            "metodo_optimizacion": descripcion_opt_usada_r.get(nombre_detalle_r, "none"),
                            "cv_desc": cv_desc_r,
                            "prop_test_desc": f"{int(st.session_state['reg_prop_test'] * 100)}% ({st.session_state.get('reg_metodo_split', 'Random')})",
                            "metricas": metricas_ficha_r,
                            "entorno_software": mu.info_entorno_software(),
                        }
                        st.session_state.modelos_guardados[nombre_guardado_r] = {
                            "tipo": "regression",
                            "modelo": res["modelo_final"],
                            "mascara_variables": mascara_variables_r,
                            "numeros_onda": st.session_state["reg_eje_usado"],
                            "pasos_pretratamiento": st.session_state["reg_pasos_pretratamiento"],
                            "ficha": ficha_r,
                            # Cross-validation RMSE, kept to build an approximate 95% prediction
                            # interval later (point prediction ± 1.96 x RMSE_cv) — a standard,
                            # model-agnostic way to report uncertainty in chemometric calibration,
                            # independent of which regression algorithm was used.
                            "rmse_cv": res["cv"]["rmse_cv"],
                        }
                        st.session_state["reg_ultima_ficha"] = ficha_r
                        st.success(f"Model '{nombre_guardado_r}' saved (traceability ID: {id_trazabilidad_r}). "
                                   "It is now available in the Prediction tab.")

                    if st.session_state.get("reg_ultima_ficha", {}).get("nombre_modelo") == nombre_guardado_r:
                        pdf_ficha_r = ru.generar_pdf_ficha_modelo(
                            st.session_state["reg_ultima_ficha"],
                            fig_extra=ru.fig_predicho_vs_real(y_t, y_p),
                            titulo_fig_extra="Predicted vs. actual",
                        )
                        st.download_button(
                            "📄 Download model card (PDF)",
                            data=bytes(pdf_ficha_r.output()),
                            file_name=f"model_card_{nombre_guardado_r}.pdf", mime="application/pdf",
                            key="descargar_ficha_reg",
                        )

# TAB: PREDICTION ON NEW SAMPLES
# -----------------------------------------------------------------------
with tabs[10]:
    st.subheader("Prediction on new samples")

    with st.expander("💾 Save/load models on your PC (to use them another day)"):
        st.caption("Models saved with the button in the Classification/Regression tabs "
                   "only live for as long as this browser session lasts. To keep them "
                   "for real, download them here as a file and upload them again when you need them.")

        if st.session_state.modelos_guardados:
            st.markdown("**Models available in this session:**")
            for nombre_m, bundle_m in st.session_state.modelos_guardados.items():
                col_a, col_b = st.columns([3, 1])
                col_a.write(f"🔹 **{nombre_m}** ({bundle_m['tipo']})")
                buffer_descarga = io.BytesIO()
                joblib.dump(bundle_m, buffer_descarga)
                col_b.download_button(
                    "⬇️ Download", data=buffer_descarga.getvalue(),
                    file_name=f"model_{nombre_m}.joblib", mime="application/octet-stream",
                    key=f"descargar_{nombre_m}",
                )

        archivo_modelo = st.file_uploader("Upload a saved model (.joblib)", type=["joblib"], key="subir_modelo")
        if archivo_modelo is not None:
            nombre_cargado = st.text_input(
                "Name for this model", value=archivo_modelo.name.replace(".joblib", ""), key="nombre_modelo_cargado",
            )
            if st.button("📤 Load this model"):
                try:
                    bundle_cargado = joblib.load(archivo_modelo)
                    claves_esperadas = {"tipo", "modelo", "mascara_variables", "numeros_onda", "pasos_pretratamiento"}
                    if not claves_esperadas.issubset(bundle_cargado.keys()):
                        st.error("The file does not have the format expected for a model saved by this app.")
                    else:
                        st.session_state.modelos_guardados[nombre_cargado] = bundle_cargado
                        st.success(f"Model '{nombre_cargado}' loaded successfully.")
                        st.rerun()
                except Exception as e:
                    st.error(f"Could not load the file: {e}")

    if not st.session_state.modelos_guardados:
        st.info("You haven't saved any model yet. Train one in the Classification or "
                 "Regression tab and use the 'Save this model' button, "
                 "or upload a previously saved one above.")
    else:
        nombre_modelo_usar = st.selectbox(
            "Model to use", list(st.session_state.modelos_guardados.keys()),
            help="Which saved model (or SIMCA model set) to apply to the new samples uploaded below.",
        )
        bundle = st.session_state.modelos_guardados[nombre_modelo_usar]
        st.caption(f"Type: {bundle['tipo']} | Preprocessing: "
                   + (" → ".join(p[0] for p in bundle["pasos_pretratamiento"]) or "none")
                   + f" | Variables used: "
                   + ("all" if bundle["mascara_variables"] is None else str(int(bundle["mascara_variables"].sum()))))

        ficha_bundle = bundle.get("ficha")
        if ficha_bundle:
            col_id, col_pdf = st.columns([3, 1])
            col_id.info(f"🔖 Traceability ID: **{ficha_bundle.get('id_trazabilidad', 'n/a')}** "
                        f"(trained on {ficha_bundle.get('fecha_creacion', 'n/a')})")
            pdf_ficha_pred = ru.generar_pdf_ficha_modelo(ficha_bundle)
            col_pdf.download_button(
                "📄 Model card (PDF)",
                data=bytes(pdf_ficha_pred.output()),
                file_name=f"model_card_{nombre_modelo_usar}.pdf", mime="application/pdf",
                key=f"descargar_ficha_pred_{nombre_modelo_usar}",
            )
            with st.expander("View full model card"):
                for clave, valor in ficha_bundle.items():
                    if clave in ("hiperparametros", "metricas", "entorno_software") and isinstance(valor, dict):
                        st.markdown(f"**{clave}:**")
                        st.json(valor)
                    else:
                        st.markdown(f"**{clave}:** {valor}")
        else:
            st.caption("This model was saved before the traceability card existed "
                       "(or it was loaded from a .joblib from an earlier version of the app) — "
                       "it has no card attached.")

        archivo_nuevo = st.file_uploader(
            "File with the new samples (same format: ID + spectra)",
            type=["csv", "xlsx", "xls"], key="archivo_prediccion",
        )
        if archivo_nuevo is not None:
            try:
                if archivo_nuevo.name.lower().endswith(".csv"):
                    df_nuevo = pd.read_csv(archivo_nuevo, index_col=0)
                else:
                    df_nuevo = pd.read_excel(archivo_nuevo, index_col=0)
                ids_nuevo = df_nuevo.index.astype(str).to_numpy()
                eje_nuevo = ejes_a_float(df_nuevo.columns.tolist())
                X_nuevo = df_nuevo.to_numpy(dtype=float)

                # 1) Interpolate onto the exact axis the model was trained with
                #    (supports the new instrument not measuring exactly the same
                #    wavenumbers / bins as during calibration).
                X_interp, fuera_de_rango = cu.interpolar_a_eje(X_nuevo, eje_nuevo, bundle["numeros_onda"])
                if fuera_de_rango:
                    st.warning("The spectral axis of the new file does not fully cover the range "
                               "used to train the model — there is extrapolation at the edges, "
                               "results there may be less reliable.")

                # 2) Apply exactly the same preprocessing (Step A/B) used during training
                X_pret_nuevo = X_interp
                for paso_tup in bundle["pasos_pretratamiento"]:
                    X_pret_nuevo = aplicar_paso(X_pret_nuevo, paso_tup)

                # 3) Apply the same variable selection (if any)
                if bundle["mascara_variables"] is not None:
                    X_final = X_pret_nuevo[:, bundle["mascara_variables"]]
                else:
                    X_final = X_pret_nuevo

                # 4) Predict
                if bundle["tipo"] == "simca":
                    modelos_simca_bundle = bundle["modelo"]  # {class_name: class-model dict}
                    matriz_dentro_pred = pd.DataFrame(
                        index=ids_nuevo, columns=list(modelos_simca_bundle.keys()), dtype=bool,
                    )
                    for c, modelo_c in modelos_simca_bundle.items():
                        _, _, dentro_c = cu.evaluar_muestras_simca(X_final, modelo_c)
                        matriz_dentro_pred[c] = dentro_c

                    def _resumen_asignacion_simca(fila):
                        aceptadas = [c for c in matriz_dentro_pred.columns if fila[c]]
                        if len(aceptadas) == 0:
                            return "None (outlier)"
                        if len(aceptadas) == 1:
                            return aceptadas[0]
                        return "Ambiguous: " + ", ".join(aceptadas)

                    df_resultado = pd.DataFrame({"id": ids_nuevo})
                    for c in matriz_dentro_pred.columns:
                        df_resultado[f"in_class_{c}"] = matriz_dentro_pred[c].to_numpy()
                    df_resultado["assignment"] = [
                        _resumen_asignacion_simca(matriz_dentro_pred.loc[i]) for i in matriz_dentro_pred.index
                    ]
                    st.dataframe(df_resultado, width='stretch')
                else:
                    predicciones = bundle["modelo"].predict(X_final)
                    if bundle["tipo"] == "classification":
                        df_resultado = pd.DataFrame({"id": ids_nuevo, "predicted_class": predicciones})
                        columnas_prob = []
                        if hasattr(bundle["modelo"], "predict_proba"):
                            try:
                                proba = bundle["modelo"].predict_proba(X_final)
                                clases_modelo = getattr(bundle["modelo"], "classes_", None)
                                if clases_modelo is not None:
                                    for j, c in enumerate(clases_modelo):
                                        nombre_col = f"probability_{c}"
                                        df_resultado[nombre_col] = proba[:, j]
                                        columnas_prob.append(nombre_col)
                                    # The probability of whichever class was actually predicted —
                                    # a quick, single "how confident is the model?" number per sample.
                                    df_resultado["confidence"] = proba.max(axis=1)
                            except Exception:
                                pass
                        st.caption("**confidence** is the model's probability for its own predicted "
                                   "class — closer to 100% means the model is more sure; values "
                                   "closer to 1/(number of classes) mean it's essentially guessing.")
                        if columnas_prob:
                            formato_cols = {c: "{:.1%}" for c in columnas_prob + ["confidence"]}
                            st.dataframe(df_resultado.style.format(formato_cols), width='stretch')
                        else:
                            st.dataframe(df_resultado, width='stretch')
                    else:
                        df_resultado = pd.DataFrame({"id": ids_nuevo, "predicted_value": predicciones})
                        rmse_cv_modelo = bundle.get("rmse_cv")
                        if rmse_cv_modelo is not None:
                            margen_95 = 1.96 * rmse_cv_modelo
                            df_resultado["uncertainty_95 (±)"] = margen_95
                            df_resultado["lower_95"] = predicciones - margen_95
                            df_resultado["upper_95"] = predicciones + margen_95
                            st.caption(
                                f"The **95% interval** (`lower_95` to `upper_95`) is an approximate "
                                f"prediction interval, built from this model's cross-validation RMSE "
                                f"(±1.96 × {rmse_cv_modelo:.4g}) — the standard way to report "
                                f"uncertainty in a chemometric calibration. It assumes the error is "
                                f"roughly similar in size across the range the model was trained on; "
                                f"treat it with extra caution for predictions near or beyond the "
                                f"edges of that range."
                            )
                        else:
                            st.caption("⚠ This model was saved before uncertainty tracking was added "
                                       "(or loaded from an older .joblib) — no 95% interval available "
                                       "for it. Retrain and save it again to get one.")
                        st.dataframe(df_resultado, width='stretch')
                st.download_button(
                    "⬇️ Download predictions",
                    data=df_resultado.to_csv(index=False).encode("utf-8"),
                    file_name=f"predictions_{nombre_modelo_usar}.csv", mime="text/csv",
                )
                st.download_button(
                    "⬇️ Download predictions (Excel)",
                    data=df_a_excel_bytes({"predictions": df_resultado}),
                    file_name=f"predictions_{nombre_modelo_usar}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="descargar_predicciones_excel",
                )

                st.divider()
                st.markdown("**📄 Report for this prediction**")
                if st.button("🖨️ Generate report (PDF)", key="generar_reporte_pred"):
                    secciones_rep_pred = [
                        {"tipo": "titulo", "texto": "1. Model used"},
                    ]
                    if ficha_bundle:
                        secciones_rep_pred.append({"tipo": "clave_valor", "pares": [
                            ("Traceability ID", ficha_bundle.get("id_trazabilidad", "n/a")),
                            ("Training date", ficha_bundle.get("fecha_creacion", "n/a")),
                            ("Algorithm", ficha_bundle.get("algoritmo", "n/a")),
                            ("Preprocessing", ficha_bundle.get("pretratamiento_desc", "n/a")),
                            ("Variables used", ficha_bundle.get("n_variables_usadas", "n/a")),
                        ]})
                    else:
                        secciones_rep_pred.append({"tipo": "clave_valor", "pares": [
                            ("Model name", nombre_modelo_usar),
                            ("Type", bundle["tipo"]),
                            ("Preprocessing", " -> ".join(p[0] for p in bundle["pasos_pretratamiento"]) or "none"),
                        ]})
                    secciones_rep_pred += [
                        {"tipo": "titulo", "texto": "2. New samples"},
                        {"tipo": "clave_valor", "pares": [
                            ("File", archivo_nuevo.name),
                            ("Number of samples", len(ids_nuevo)),
                        ]},
                        {"tipo": "titulo", "texto": "3. Predictions"},
                        {"tipo": "tabla",
                         "encabezados": list(df_resultado.columns),
                         "filas": df_resultado.astype(str).values.tolist()[:80]},
                    ]
                    if len(df_resultado) > 80:
                        secciones_rep_pred.append({"tipo": "parrafo",
                            "texto": f"(showing the first 80 of {len(df_resultado)} predictions — "
                                     "the full table is in the Excel file downloadable from the app)"})
                    pdf_reporte_pred = ru.generar_reporte(
                        "Prediction Report", f"Model: {nombre_modelo_usar}", secciones_rep_pred,
                    )
                    st.download_button(
                        "⬇️ Download full report (PDF)",
                        data=bytes(pdf_reporte_pred.output()),
                        file_name="prediction_report.pdf", mime="application/pdf",
                        key="descargar_reporte_pred",
                    )
            except Exception as e:
                st.error(f"Could not predict on the new file: {e}")

        st.divider()
        if st.button("🗑️ Delete all saved models"):
            st.session_state.modelos_guardados = {}
            st.rerun()
