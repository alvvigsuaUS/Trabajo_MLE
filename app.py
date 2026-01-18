# app.py
# Streamlit (formulario) + ngrok opcional + Clasificador (probabilidades) + Regresor (nota)
#
# Cambios aplicados:
# 1) Rangos de clases:
#    failed [0,49], passed [50,59], good [60,69], merit [70,89], excellent [90,100]
# 2) Explicación simple:
#    - (1) cómo se calcula la nota sugerida por el clasificador: sum(p * medio)
#    - (2) sigma del clasificador: SOLO se muestra el valor (sin explicar cálculo)
#    - (3) ajuste por intervalos según |delta| (sin fusión)
# 3) Se elimina la explicación 4 y se elimina la explicación del cálculo de sigma.

import os
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import joblib

# =========================
# RUTAS
# =========================
CLASSIFIER_PATH = "models/classifier.keras"
REGRESSOR_PATH  = "models/best_model.keras"
X_PREPROCESS_PATH = "artifacts/x_preprocess.joblib"
Y_SCALER_PATH     = "artifacts/y_scaler.joblib"

CLASS_NAMES = ["failed", "passed", "good", "merit", "excellent"]

CLASS_RANGES = {
    "failed": (0.0, 49.0),
    "passed": (50.0, 59.0),
    "good": (60.0, 69.0),
    "merit": (70.0, 89.0),
    "excellent": (90.0, 100.0),
}

REG_ERROR_POINTS = 10.0
EPS = 1e-9


# =========================
# UTILIDADES (clasificador -> nota esperada + sigma)
# =========================
def _mid_and_var_uniform(lo: float, hi: float):
    lo = float(lo); hi = float(hi)
    mu = (lo + hi) / 2.0
    var = ((hi - lo) ** 2) / 12.0
    return mu, var

def class_stats_and_table_from_probs(probs, class_names=CLASS_NAMES, class_ranges=CLASS_RANGES):
    p = np.asarray(probs, dtype=float)
    p = p / (p.sum() + EPS)

    rows = []
    mus = []
    vars_ = []

    for i, c in enumerate(class_names):
        lo, hi = class_ranges[c]
        mu_c, var_c = _mid_and_var_uniform(lo, hi)
        mus.append(mu_c)
        vars_.append(var_c)

        rows.append({
            "clase": c,
            "p(clase)": float(p[i]),
            "rango": f"[{lo:.0f}, {hi:.0f}]",
            "medio(mu_c)": float(mu_c),
            "p*mu_c": float(p[i] * mu_c),
        })

    mus = np.array(mus, dtype=float)
    vars_ = np.array(vars_, dtype=float)

    mu_cls = float(np.sum(p * mus))

    var_cls = float(np.sum(p * (vars_ + mus**2)) - mu_cls**2)
    var_cls = max(var_cls, EPS)
    sigma_cls = float(np.sqrt(var_cls))

    df = pd.DataFrame(rows)
    return mu_cls, sigma_cls, df


# =========================
# AJUSTE POR INTERVALOS (pedido)
# =========================
ADJUSTMENT_INTERVALS = [
    (0,   3,   0),
    (3,   6,   2),
    (6,  10,   4),
    (10, 15,   6),
    (15, 20,   8),
    (20, float("inf"), 10),
]

def interval_adjustment(delta, intervals=ADJUSTMENT_INTERVALS):
    abs_d = abs(float(delta))
    sign = 1.0 if delta >= 0 else -1.0

    chosen = None
    for a, b, step in intervals:
        if a <= abs_d < b:
            chosen = (a, b, step)
            break
    if chosen is None:
        chosen = intervals[-1]

    a, b, step = chosen
    adj = sign * float(step)

    if b == float("inf"):
        label = f"|delta| >= {a}"
    else:
        label = f"{a} <= |delta| < {b}"

    return adj, label, abs_d


# =========================
# PREDICCIONES
# =========================
def predict_classifier_probs(clf, X_scaled_np):
    probs = clf.predict(X_scaled_np, verbose=0)
    probs = np.asarray(probs)
    if probs.ndim == 1:
        probs = probs.reshape(1, -1)
    return probs[0]

def predict_regressor_scaled(reg, X_scaled_np):
    y = reg.predict(X_scaled_np, verbose=0)
    y = np.asarray(y).reshape(-1)
    return float(y[0])


# =========================
# CARGA (modelos + scalers)
# =========================
@st.cache_resource
def load_all():
    clf = tf.keras.models.load_model(CLASSIFIER_PATH)
    reg = tf.keras.models.load_model(REGRESSOR_PATH)

    x_scaler = joblib.load(X_PREPROCESS_PATH)
    y_scaler = joblib.load(Y_SCALER_PATH) if os.path.exists(Y_SCALER_PATH) else None

    if not hasattr(x_scaler, "feature_names_in_"):
        raise ValueError("x_preprocess.joblib debe estar fitteado con un DataFrame (necesita feature_names_in_).")

    cols = list(x_scaler.feature_names_in_)
    data_min = np.asarray(getattr(x_scaler, "data_min_", None), dtype=float)
    data_max = np.asarray(getattr(x_scaler, "data_max_", None), dtype=float)
    if data_min is None or data_max is None:
        raise ValueError("El X_scaler no tiene data_min_/data_max_.")

    return clf, reg, x_scaler, y_scaler, cols, data_min, data_max


# =========================
# NGROK (opcional)
# =========================
@st.cache_resource
def start_ngrok():
    token = os.getenv("NGROK_AUTHTOKEN", "").strip()
    if not token:
        return None
    try:
        from pyngrok import ngrok, conf
        conf.get_default().auth_token = token
        tunnels = ngrok.get_tunnels()
        for t in tunnels:
            if "8501" in t.public_url:
                return t.public_url
        return ngrok.connect(8501, "http").public_url
    except Exception:
        return None


# =========================
# UI
# =========================
st.set_page_config(page_title="Predicción de nota (clasificador + regresor)", layout="centered")

st.title("Predicción de nota: Clasificador (probabilidades) + Regresor (nota)")
st.write(
    "El clasificador aporta probabilidades por clase; con ellas se calcula una nota sugerida por rangos. "
    "El regresor aporta una nota. Como el regresor tiene un error típico de ±10 puntos, "
    "la corrección se realiza por intervalos: si la diferencia es pequeña se corrige poco; si es grande se corrige más."
)

with st.sidebar:
    st.header("Acceso público (ngrok)")
    url = start_ngrok()
    if url:
        st.success(f"URL pública: {url}")
    else:
        st.info("Si no aparece URL, ejecutar ngrok aparte: `ngrok http 8501` o definir NGROK_AUTHTOKEN.")

    st.header("Parámetros")
    st.write(f"Error típico del regresor: ±{REG_ERROR_POINTS:.0f} puntos.")
    st.caption("La corrección máxima se limita a ±10 (margen del error del regresor).")

# Cargar recursos
try:
    clf, reg, x_scaler, y_scaler, cols, data_min, data_max = load_all()
except Exception as e:
    st.error("No se pudieron cargar modelos o artifacts. Revisar rutas y archivos.")
    st.code(str(e))
    st.stop()

defaults = (data_min + data_max) / 2.0

st.subheader("Formulario de entrada (1 estudiante)")
st.caption(
    "Los rangos de cada variable se obtienen de los mínimos y máximos que vio el MinMaxScaler al entrenar. "
    "Si se detecta variable binaria o discreta, se muestra como selector."
)

def is_close_to_int(x, tol=1e-6):
    return abs(float(x) - round(float(x))) <= tol

def build_input_widget(name, lo, hi, default):
    lo_f, hi_f, def_f = float(lo), float(hi), float(default)

    # Binaria 0/1
    if abs(lo_f - 0.0) < 1e-9 and abs(hi_f - 1.0) < 1e-9:
        choice = st.selectbox(label=name, options=[0, 1], index=1 if def_f >= 0.5 else 0)
        return float(choice)

    # Discreta pequeña
    if is_close_to_int(lo_f) and is_close_to_int(hi_f):
        span = int(round(hi_f - lo_f))
        if 0 <= span <= 12:
            opts = list(range(int(round(lo_f)), int(round(hi_f)) + 1))
            closest = min(range(len(opts)), key=lambda i: abs(opts[i] - def_f))
            choice = st.selectbox(label=name, options=opts, index=closest)
            return float(choice)

    # Continua
    step = (hi_f - lo_f) / 100.0 if hi_f > lo_f else 0.01
    step = max(step, 0.01)

    value = st.number_input(
        label=name,
        min_value=lo_f,
        max_value=hi_f,
        value=float(np.clip(def_f, lo_f, hi_f)),
        step=step,
        help=f"Rango visto en entrenamiento: [{lo_f:.3f}, {hi_f:.3f}]"
    )
    return float(value)

with st.form("student_form", clear_on_submit=False):
    col_left, col_right = st.columns(2)
    inputs = {}
    for i, c in enumerate(cols):
        lo, hi, d = data_min[i], data_max[i], defaults[i]
        target_col = col_left if i % 2 == 0 else col_right
        with target_col:
            inputs[c] = build_input_widget(c, lo, hi, d)

    submitted = st.form_submit_button("Predecir")

if not submitted:
    st.stop()

# DataFrame 1 fila en orden correcto
df_one = pd.DataFrame([{c: inputs[c] for c in cols}], columns=cols)

# Transformar X con el scaler
try:
    X_scaled = x_scaler.transform(df_one)
    if isinstance(X_scaled, pd.DataFrame):
        X_scaled_np = X_scaled.to_numpy(dtype=np.float32)
    else:
        X_scaled_np = np.asarray(X_scaled, dtype=np.float32)
except Exception as e:
    st.error("Error transformando la entrada con X_scaler.")
    st.code(str(e))
    st.stop()

# Predicción del clasificador (probabilidades)
probs = predict_classifier_probs(clf, X_scaled_np)

# Predicción del regresor (escala interna)
y_reg_scaled = predict_regressor_scaled(reg, X_scaled_np)

# Volver a escala 0-100 si hay y_scaler
try:
    if y_scaler is not None:
        y_reg_0_100 = float(y_scaler.inverse_transform(np.array([[y_reg_scaled]], dtype=float))[0, 0])
    else:
        y_reg_0_100 = float(y_reg_scaled)
except Exception:
    y_reg_0_100 = float(y_reg_scaled)

y_reg_0_100 = float(np.clip(y_reg_0_100, 0.0, 100.0))

# Nota sugerida por clasificador (mu_cls) + sigma + tabla de cuentas
mu_cls, sigma_cls, df_calc = class_stats_and_table_from_probs(probs)

# Diferencia y ajuste por intervalos
delta = float(mu_cls - y_reg_0_100)
adj, interval_label, abs_d = interval_adjustment(delta)
note_recommended = float(np.clip(y_reg_0_100 + adj, 0.0, 100.0))

pred_class = int(np.argmax(probs))
pred_name = CLASS_NAMES[pred_class]
pred_conf = float(probs[pred_class])

# =========================
# SALIDA PRINCIPAL
# =========================
st.subheader("Resultados")
st.markdown(f"**Clase más probable:** `{pred_name}` (p = {pred_conf:.4f})")
st.markdown(f"**Nota del regresor:** {y_reg_0_100:.2f} / 100 (error típico ±{REG_ERROR_POINTS:.0f})")
st.markdown(f"**Nota sugerida por el clasificador (por rangos):** {mu_cls:.2f} / 100")
st.markdown(f"**Sigma del clasificador (valor):** {sigma_cls:.2f}")

st.markdown(f"**Diferencia:** delta = (clasificador - regresor) = **{delta:.2f}** puntos")

if adj > 0:
    st.success(f"**Ajuste por intervalos:** {interval_label} -> se suma **+{adj:.0f}** puntos")
elif adj < 0:
    st.warning(f"**Ajuste por intervalos:** {interval_label} -> se resta **{abs(adj):.0f}** puntos")
else:
    st.info(f"**Ajuste por intervalos:** {interval_label} -> **no se ajusta** (0 puntos)")

st.markdown(f"**Nota recomendada (regresor ajustado):** {note_recommended:.2f} / 100")

with st.expander("Probabilidades por clase"):
    st.dataframe(pd.DataFrame({"clase": CLASS_NAMES, "probabilidad": probs}), use_container_width=True)

with st.expander("Entrada usada (1 fila)"):
    st.dataframe(df_one, use_container_width=True)

#Explicación
with st.expander("Explicación simple (con cuentas)", expanded=True):
    st.markdown("### 1) De probabilidades a nota sugerida")
    st.markdown(
        "Cada clase corresponde a un rango de notas. Se usa el punto medio del rango como nota típica y se hace:\n\n"
        "**nota_clasificador = sum( p(clase) × medio(clase) )**"
    )
    st.dataframe(df_calc[["clase", "p(clase)", "rango", "medio(mu_c)", "p*mu_c"]], use_container_width=True)
    st.markdown(f"**Suma p×medio = {float(df_calc['p*mu_c'].sum()):.4f} → nota_clasificador = {mu_cls:.2f}**")

    st.markdown("### 2) Sigma del clasificador")
    st.markdown(f"**sigma_clasificador = {sigma_cls:.2f}**")

    st.markdown("### 3) Ajuste por intervalos según la diferencia")
    st.markdown(
        "Se calcula:\n\n"
        "**delta = nota_clasificador - nota_regresor**\n\n"
        "Según el tamaño de |delta|, se ajusta poco a poco, con un máximo de ±10 puntos:"
    )

    df_intervals = pd.DataFrame(
        [{"intervalo": f"{a} <= |delta| < {b if b != float('inf') else 'inf'}", "ajuste (puntos)": step}
         for a, b, step in ADJUSTMENT_INTERVALS]
    )
    st.dataframe(df_intervals, use_container_width=True)

    st.markdown(
        f"- nota_regresor = **{y_reg_0_100:.2f}**\n"
        f"- nota_clasificador = **{mu_cls:.2f}**\n"
        f"- delta = **{delta:.2f}** (|delta| = {abs_d:.2f})\n"
        f"- intervalo elegido: **{interval_label}**\n"
        f"- ajuste aplicado: **{adj:+.0f}** puntos\n"
        f"- nota recomendada: **{note_recommended:.2f}**"
    )
