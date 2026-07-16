import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import cv2
import keras
import random
from streamlit_drawable_canvas import st_canvas

# ── Configuración de la página ────────────────────────────────────────────────
st.set_page_config(
    page_title="Reconocedor de Dígitos",
    page_icon="✏️",
    layout="wide"
)

# ── Cargar el modelo ──────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return keras.models.load_model("model.keras")

model = load_model()

# ── Modelo de activación ──────────────────────────────────────────────────────
@st.cache_resource
def load_activation_model():
    full_model = load_model()
    dummy_input = np.zeros((1, 28, 28, 1), dtype="float32")
    full_model.predict(dummy_input, verbose=0)
    conv1_layer = full_model.layers[0]
    conv2_layer = full_model.layers[2]
    return keras.Model(
        inputs=full_model.inputs,
        outputs=[conv1_layer.output, conv2_layer.output]
    )

try:
    activation_model = load_activation_model()
    activation_available = True
except Exception:
    activation_model = None
    activation_available = False

# ── Nombres de las clases ─────────────────────────────────────────────────────
CLASS_NAMES = {
    0: "Cero", 1: "Uno", 2: "Dos", 3: "Tres", 4: "Cuatro",
    5: "Cinco", 6: "Seis", 7: "Siete", 8: "Ocho", 9: "Nueve"
}

# ── Inicializar session_state ─────────────────────────────────────────────────
defaults = {
    "mode": "Normal",
    "challenge_score": 0,
    "challenge_best": 0,
    "challenge_target": random.randint(0, 9),
    "challenge_result": None,
    "canvas_key": "canvas_0",
    "canvas_counter": 0
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Función de preprocesamiento ───────────────────────────────────────────────
@st.cache_data
def preprocess_image(image_data, threshold, apply_blur, invert):
    img = Image.fromarray(image_data.astype("uint8")).convert("L")
    if invert:
        img = ImageOps.invert(img)
    img_array = np.array(img)
    if apply_blur:
        img_array = cv2.GaussianBlur(img_array, (5, 5), 0)
    img_array = np.where(img_array > threshold, 255, 0).astype("uint8")
    img_resized = Image.fromarray(img_array).resize((28, 28))
    img_final = np.array(img_resized).astype("float32") / 255.0
    img_final = img_final.reshape(1, 28, 28, 1)
    return img_final, img_resized

# ── Función de mapas de activación ───────────────────────────────────────────
def show_activation_maps(img_final, img_resized):
    st.subheader("Mapas de activación de la CNN")
    st.caption("Visualización de lo que detecta cada capa convolucional en tu dibujo.")
    conv1_acts, conv2_acts = activation_model.predict(img_final, verbose=0)
    conv1_mean = conv1_acts[0].mean(axis=-1)
    conv2_mean = conv2_acts[0].mean(axis=-1)

    def normalize_map(arr):
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max - arr_min > 0:
            arr = (arr - arr_min) / (arr_max - arr_min) * 255
        return arr.astype("uint8")

    conv1_img = Image.fromarray(normalize_map(conv1_mean)).resize((112, 112))
    conv2_img = Image.fromarray(normalize_map(conv2_mean)).resize((112, 112))

    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption("Capa 1: bordes")
        st.image(conv1_img, use_container_width=True)
    with col2:
        st.caption("Capa 2: patrones")
        st.image(conv2_img, use_container_width=True)
    with col3:
        st.caption("Input (28×28)")
        st.image(img_resized, use_container_width=True)

# ── Título ────────────────────────────────────────────────────────────────────
st.title("Reconocedor de Dígitos")
st.markdown("""
App de reconocimiento de dígitos escritos a mano usando una
**red neuronal convolucional (CNN)** entrenada con MNIST.
""")
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Modo")

mode = st.sidebar.radio(
    "Selecciona el modo",
    options=["Normal", "Desafío"],
    index=0 if st.session_state.mode == "Normal" else 1,
    label_visibility="collapsed"
)

if mode != st.session_state.mode:
    st.session_state.mode = mode
    st.session_state.canvas_counter += 1
    st.session_state.canvas_key = f"canvas_{st.session_state.canvas_counter}"
    st.rerun()

st.sidebar.divider()
st.sidebar.header("Parámetros de preprocesamiento")

stroke_width = st.sidebar.slider(
    "Grosor del trazo", min_value=5, max_value=30, value=15,
    help="Controla el grosor del trazo en el canvas"
)
threshold = st.sidebar.slider(
    "Umbral de binarización", min_value=0, max_value=255, value=128,
    help="Píxeles por encima de este valor se activan como blancos"
)
apply_blur = st.sidebar.checkbox(
    "Aplicar suavizado gaussiano", value=True,
    help="Suaviza el trazo antes de predecir"
)
invert_colors = st.sidebar.toggle(
    "Invertir colores", value=False,
    help="MNIST espera dígito blanco sobre fondo negro"
)
st.sidebar.divider()
if st.sidebar.button("Limpiar canvas"):
    st.session_state.canvas_counter += 1
    st.session_state.canvas_key = f"canvas_{st.session_state.canvas_counter}"
    st.rerun()

# ════════════════════════════════════════════════════════════
# MODO DESAFÍO: marcador e instrucción
# ════════════════════════════════════════════════════════════
if st.session_state.mode == "Desafío":

    col_score1, col_score2 = st.columns(2)
    with col_score1:
        st.metric("Racha actual", st.session_state.challenge_score)
    with col_score2:
        st.metric("Mejor racha", st.session_state.challenge_best)

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #1a1a2e, #16213e);
            border: 2px solid #00d4aa;
            border-radius: 16px;
            padding: 24px;
            text-align: center;
            margin-bottom: 24px;
        ">
            <div style="font-size: 14px; color: #aaaaaa;
                 letter-spacing: 2px; text-transform: uppercase;
                 margin-bottom: 8px;">Dibuja este dígito</div>
            <div style="font-size: 140px; font-weight: bold; color: #00d4aa;
                 font-family: monospace; line-height: 1;">
                {st.session_state.challenge_target}
            </div>
            <div style="font-size: 20px; color: #ffffff; margin-top: 8px;
                 letter-spacing: 3px; text-transform: uppercase;">
                {CLASS_NAMES[st.session_state.challenge_target]}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.divider()

# ════════════════════════════════════════════════════════════
# CANVAS: único para ambos modos
# ════════════════════════════════════════════════════════════
col_canvas, col_result = st.columns([1, 1], gap="large")

with col_canvas:
    st.subheader("Dibuja aquí")

    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=stroke_width,
        stroke_color="white",
        background_color="black",
        height=280, width=280,
        drawing_mode="freedraw",
        key=st.session_state.canvas_key
    )

    if canvas_result.image_data is not None:
        if canvas_result.image_data[:, :, 0].sum() > 0:
            _, img_resized_preview = preprocess_image(
                canvas_result.image_data, threshold, apply_blur, invert_colors
            )
            st.caption("Lo que ve el modelo (28×28):")
            st.image(img_resized_preview, width=100)

with col_result:

    # ── MODO NORMAL ──────────────────────────────────────────
    if st.session_state.mode == "Normal":
        st.subheader("Resultado")

        if canvas_result.image_data is not None and canvas_result.image_data[:, :, 0].sum() > 0:

            img_final, img_resized = preprocess_image(
                canvas_result.image_data, threshold, apply_blur, invert_colors
            )
            prediction = model.predict(img_final, verbose=0)
            predicted_class = int(np.argmax(prediction))
            confidence = float(np.max(prediction)) * 100
            nombre = CLASS_NAMES[predicted_class]

            if confidence >= 70:
                st.markdown(
                    f"""
                    <div style="
                        background: linear-gradient(135deg, #1a1a2e, #16213e);
                        border: 2px solid #00d4aa;
                        border-radius: 16px;
                        padding: 32px;
                        text-align: center;
                        margin-bottom: 16px;
                    ">
                        <div style="font-size: 120px; font-weight: bold; color: #00d4aa;
                             font-family: monospace; line-height: 1;">{predicted_class}</div>
                        <div style="font-size: 32px; font-weight: bold; color: #ffffff;
                             margin-top: 12px; letter-spacing: 4px;
                             text-transform: uppercase;">{nombre}</div>
                        <div style="font-size: 16px; color: #aaaaaa;
                             margin-top: 12px;">Confianza: {confidence:.1f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div style="
                        background: linear-gradient(135deg, #2e1a1a, #3e1616);
                        border: 2px solid #ffaa00;
                        border-radius: 16px;
                        padding: 32px;
                        text-align: center;
                        margin-bottom: 16px;
                    ">
                        <div style="font-size: 120px; font-weight: bold; color: #ffaa00;
                             font-family: monospace; line-height: 1;">{predicted_class}</div>
                        <div style="font-size: 32px; font-weight: bold; color: #ffffff;
                             margin-top: 12px; letter-spacing: 4px;
                             text-transform: uppercase;">{nombre}</div>
                        <div style="font-size: 16px; color: #ffaa00; margin-top: 12px;">
                            Confianza baja: {confidence:.1f}% — intenta dibujar más claro</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.progress(int(confidence))

            st.divider()
            st.subheader("Top 3 predicciones")
            top3_idx = np.argsort(prediction[0])[::-1][:3]
            for i, idx in enumerate(top3_idx):
                prob = float(prediction[0][idx]) * 100
                medal = ["🥇", "🥈", "🥉"][i]
                st.markdown(
                    f"""
                    <div style="
                        background: #1a1a2e;
                        border-radius: 8px;
                        padding: 12px 16px;
                        margin-bottom: 8px;
                    ">
                        <span style="font-size: 20px; margin-right: 12px;">{medal}</span>
                        <span style="font-size: 28px; font-weight: bold; color: #ffffff;
                              font-family: monospace; margin-right: 12px;">{idx}</span>
                        <span style="font-size: 16px; color: #aaaaaa;
                              margin-right: 8px;">{CLASS_NAMES[idx]}</span>
                        <span style="font-size: 16px; color: #00d4aa;
                              font-weight: bold; float: right;">{prob:.1f}%</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.divider()
            st.subheader("Probabilidades por clase")
            probs = {
                f"{i} — {CLASS_NAMES[i]}": float(prediction[0][i]) * 100
                for i in range(10)
            }
            st.bar_chart(probs)

            with st.expander("Ver valores exactos"):
                for clase, prob in probs.items():
                    st.write(f"{clase}: **{prob:.2f}%**")

            if activation_available:
                st.divider()
                show_activation_maps(img_final, img_resized)

        else:
            st.markdown(
                """
                <div style="
                    background: #1a1a2e;
                    border: 2px dashed #444444;
                    border-radius: 16px;
                    padding: 48px;
                    text-align: center;
                    color: #666666;
                ">
                    <div style="font-size: 16px; letter-spacing: 2px;">
                        DIBUJA UN DÍGITO PARA VER LA PREDICCIÓN
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # ── MODO DESAFÍO ─────────────────────────────────────────
    else:
        st.subheader("¿Lo reconoció?")

        if canvas_result.image_data is not None and canvas_result.image_data[:, :, 0].sum() > 0:

            img_final_ch, img_resized_ch = preprocess_image(
                canvas_result.image_data, threshold, apply_blur, invert_colors
            )
            prediction_ch = model.predict(img_final_ch, verbose=0)
            predicted_ch = int(np.argmax(prediction_ch))
            confidence_ch = float(np.max(prediction_ch)) * 100
            correct = predicted_ch == st.session_state.challenge_target

            if correct:
                st.markdown(
                    f"""
                    <div style="
                        background: linear-gradient(135deg, #0a2e2a, #0a3d35);
                        border: 2px solid #00d4aa;
                        border-radius: 16px;
                        padding: 32px;
                        text-align: center;
                    ">
                        <div style="font-size: 48px; margin-bottom: 8px;">✓</div>
                        <div style="font-size: 28px; font-weight: bold; color: #00d4aa;
                             letter-spacing: 3px;">CORRECTO</div>
                        <div style="font-size: 16px; color: #aaaaaa; margin-top: 8px;">
                            Reconoció el {predicted_ch} con {confidence_ch:.1f}% de confianza
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                if st.button("Siguiente dígito", use_container_width=True, type="primary"):
                    st.session_state.challenge_score += 1
                    if st.session_state.challenge_score > st.session_state.challenge_best:
                        st.session_state.challenge_best = st.session_state.challenge_score
                    st.session_state.challenge_target = random.randint(0, 9)
                    st.session_state.canvas_counter += 1
                    st.session_state.canvas_key = f"canvas_{st.session_state.canvas_counter}"
                    st.rerun()

            else:
                st.markdown(
                    f"""
                    <div style="
                        background: linear-gradient(135deg, #2e0a0a, #3d0a0a);
                        border: 2px solid #cc3333;
                        border-radius: 16px;
                        padding: 32px;
                        text-align: center;
                    ">
                        <div style="font-size: 48px; margin-bottom: 8px;">✗</div>
                        <div style="font-size: 28px; font-weight: bold; color: #cc3333;
                             letter-spacing: 3px;">INCORRECTO</div>
                        <div style="font-size: 16px; color: #aaaaaa; margin-top: 8px;">
                            El modelo vio un {predicted_ch} ({CLASS_NAMES[predicted_ch]}),
                            no un {st.session_state.challenge_target}
                        </div>
                        <div style="font-size: 14px; color: #888888; margin-top: 4px;">
                            Racha perdida: {st.session_state.challenge_score}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                col_retry, col_next = st.columns(2)
                with col_retry:
                    if st.button("Intentar de nuevo", use_container_width=True):
                        st.session_state.canvas_counter += 1
                        st.session_state.canvas_key = f"canvas_{st.session_state.canvas_counter}"
                        st.rerun()
                with col_next:
                    if st.button("Nuevo dígito", use_container_width=True):
                        st.session_state.challenge_score = 0
                        st.session_state.challenge_target = random.randint(0, 9)
                        st.session_state.canvas_counter += 1
                        st.session_state.canvas_key = f"canvas_{st.session_state.canvas_counter}"
                        st.rerun()

            if activation_available:
                st.divider()
                show_activation_maps(img_final_ch, img_resized_ch)

        else:
            st.markdown(
                """
                <div style="
                    background: #1a1a2e;
                    border: 2px dashed #444444;
                    border-radius: 16px;
                    padding: 48px;
                    text-align: center;
                    color: #666666;
                ">
                    <div style="font-size: 16px; letter-spacing: 2px;">
                        DIBUJA EL DÍGITO PARA VERIFICAR
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

# ── Acerca del modelo ─────────────────────────────────────────────────────────
st.divider()
with st.expander("Acerca del modelo"):
    st.markdown("""
    ### Arquitectura
    Red neuronal convolucional (CNN) entrenada con el dataset MNIST.

    | Capa | Tipo | Detalles |
    |---|---|---|
    | 1 | Conv2D | 32 filtros, kernel 3×3, activación ReLU |
    | 2 | MaxPooling2D | ventana 2×2 |
    | 3 | Conv2D | 64 filtros, kernel 3×3, activación ReLU |
    | 4 | MaxPooling2D | ventana 2×2 |
    | 5 | Flatten | aplana el tensor a vector de 1,600 valores |
    | 6 | Dense | 128 neuronas, activación ReLU |
    | 7 | Dropout | 50% de neuronas desactivadas durante entrenamiento |
    | 8 | Dense | 10 neuronas, activación Softmax |

    ### Métricas finales
    | Métrica | Valor |
    |---|---|
    | Test accuracy | 99.21% |
    | Test loss | 0.0244 |
    | Parámetros entrenables | 225,034 |
    """)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Proyecto de portafolio — Red neuronal CNN entrenada con Keras sobre el dataset MNIST.")

