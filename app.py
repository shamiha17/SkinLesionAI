import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# -------------------- CONFIG --------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_mobilenetv2_8class.keras"
LABELS_PATH = BASE_DIR / "skinlesion_8class_labels.json"
IMG_SIZE = 160

st.set_page_config(
    page_title="SkinLesionAI | AI Skin Scanner",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# -------------------- CUSTOM DESIGN --------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        --navy: #07111F;
        --panel: #0D1B2A;
        --panel2: #112438;
        --teal: #2DD4BF;
        --muted: #9CB0C5;
        --border: rgba(148, 180, 205, 0.17);
    }

    .stApp {
        background:
            radial-gradient(ellipse at 10% 0%, rgba(20,184,166,.12), transparent 35%),
            radial-gradient(ellipse at 95% 15%, rgba(14,116,144,.12), transparent 30%),
            var(--navy);
        color: #F1F5F9;
        font-family: 'DM Sans', sans-serif;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 1.6rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, .brand, .hero-title {
        font-family: 'Space Grotesk', sans-serif !important;
        color: #F1F5F9 !important;
    }

    p, li, label {
        color: #C3D1DF;
    }

    [data-testid="stFileUploader"] {
        background: rgba(13, 27, 42, .78);
        border: 1px dashed rgba(45,212,191,.55);
        border-radius: 18px;
        padding: 14px;
    }

    [data-testid="stFileUploader"] section {
        background: transparent;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: rgba(17, 36, 56, .7);
        border: 1px dashed rgba(45,212,191,.35);
        border-radius: 14px;
    }

    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #112438, #0B1928);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 18px 20px;
    }

    [data-testid="stMetricLabel"] {
        color: #9CB0C5 !important;
    }

    [data-testid="stMetricValue"] {
        color: #2DD4BF !important;
        font-family: 'Space Grotesk', sans-serif;
    }

    .brand-row {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 8px 0 22px 0;
        border-bottom: 1px solid var(--border);
        margin-bottom: 32px;
    }

    .brand-icon {
        background: rgba(45,212,191,.12);
        border: 1px solid rgba(45,212,191,.28);
        border-radius: 13px;
        width: 48px;
        height: 48px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 25px;
    }

    .brand {
        font-size: 23px;
        font-weight: 700;
        letter-spacing: -.8px;
    }

    .brand span {
        color: var(--teal);
    }

    .brand-sub {
        color: var(--muted);
        font-size: 12px;
        margin-top: 2px;
    }

    .status-pill {
        margin-left: auto;
        color: #5EEAD4;
        border: 1px solid rgba(45,212,191,.3);
        background: rgba(45,212,191,.08);
        border-radius: 30px;
        padding: 7px 12px;
        font-size: 12px;
        white-space: nowrap;
    }

    .hero {
        background: linear-gradient(120deg, rgba(17,36,56,.98), rgba(8,28,43,.92));
        border: 1px solid var(--border);
        border-radius: 24px;
        padding: 30px;
        margin-bottom: 25px;
        position: relative;
        overflow: hidden;
    }

    .eyebrow {
        color: var(--teal);
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 12px;
    }

    .hero-title {
        font-size: clamp(29px, 4vw, 43px);
        line-height: 1.12;
        letter-spacing: -1.4px;
        font-weight: 700;
        margin: 0 0 13px 0;
    }

    .hero-title span {
        color: var(--teal);
    }

    .hero-desc {
        color: #AFC1D2;
        line-height: 1.8;
        max-width: 650px;
        font-size: 15px;
    }

    .hero-chip {
        display: inline-block;
        margin-top: 15px;
        margin-right: 7px;
        padding: 7px 11px;
        border-radius: 9px;
        border: 1px solid var(--border);
        color: #C7D8E8;
        font-size: 12px;
        background: rgba(255,255,255,.025);
    }

    .section-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 21px;
        font-weight: 600;
        margin: 8px 0 5px 0;
        color: #F1F5F9;
    }

    .section-desc {
        color: var(--muted);
        font-size: 13px;
        margin-bottom: 18px;
    }

    .panel {
        background: linear-gradient(145deg, rgba(13,27,42,.98), rgba(10,24,38,.98));
        border: 1px solid var(--border);
        border-radius: 19px;
        padding: 22px;
        margin-bottom: 18px;
    }

    .panel-heading {
        font-family: 'Space Grotesk', sans-serif;
        color: #EAF2F8;
        font-size: 17px;
        font-weight: 600;
        margin-bottom: 5px;
    }

    .panel-sub {
        color: var(--muted);
        font-size: 12px;
        line-height: 1.7;
        margin-bottom: 15px;
    }

    .result-box {
        background: linear-gradient(135deg, rgba(13,148,136,.15), rgba(13,27,42,.8));
        border: 1px solid rgba(45,212,191,.35);
        border-radius: 18px;
        padding: 23px;
        margin: 12px 0 20px 0;
    }

    .result-label {
        font-size: 12px;
        color: var(--teal);
        letter-spacing: 1.4px;
        text-transform: uppercase;
        font-weight: 700;
    }

    .result-name {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 31px;
        font-weight: 700;
        margin: 8px 0;
        color: #F1F5F9;
        overflow-wrap: anywhere;
    }

    .result-note {
        color: #AFC1D2;
        font-size: 13px;
        line-height: 1.7;
    }

    div.stButton > button {
        width: 100%;
        min-height: 48px;
        border-radius: 12px;
        background: linear-gradient(100deg, #0D9488, #14B8A6);
        color: #04131B;
        border: none;
        font-weight: 700;
        transition: .2s ease;
    }

    div.stButton > button:hover {
        background: linear-gradient(100deg, #14B8A6, #5EEAD4);
        color: #04131B;
        border: none;
        transform: translateY(-1px);
    }

    [data-testid="stImage"] img {
        border-radius: 14px;
    }

    [data-testid="stAlert"] {
        border-radius: 13px;
    }

    hr {
        border-color: var(--border);
    }

    .footer {
        border-top: 1px solid var(--border);
        padding-top: 20px;
        margin-top: 35px;
        color: #8095A9;
        font-size: 12px;
        line-height: 1.8;
        text-align: center;
    }

    @media (max-width: 640px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
        .hero {
            padding: 21px;
        }
        .status-pill {
            display: none;
        }
        .brand {
            font-size: 20px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


@st.cache_data
def load_labels():
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# -------------------- HEADER --------------------
st.markdown(
    """
    <div class="brand-row">
        <div class="brand-icon">🔬</div>
        <div>
            <div class="brand">SkinLesion<span>AI</span></div>
            <div class="brand-sub">AI SKIN IMAGE CLASSIFICATION · RESEARCH DEMO</div>
        </div>
        <div class="status-pill">● Model dashboard</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">✦ AI-powered image analysis</div>
        <div class="hero-title">A closer look at your skin.<br><span>Powered by AI.</span></div>
        <div class="hero-desc">
            Upload a skin image to explore the predictions of a trained
            MobileNetV2 image-classification model. Built as an educational
            machine-learning project.
        </div>
        <span class="hero-chip">✦ MobileNetV2</span>
        <span class="hero-chip">8 output classes</span>
        <span class="hero-chip">Research prototype</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# Keep the safety notice prominent and honest.
st.warning(
    "Medical safety notice: This is an educational research demo, not a "
    "medical device. Its output cannot diagnose or rule out skin disease. "
    "Do not use it to make treatment decisions; consult a qualified healthcare professional."
)

# -------------------- FILE CHECK --------------------
if not MODEL_PATH.exists() or not LABELS_PATH.exists():
    st.error(
        "Model or labels file is missing. Keep these files beside app.py:\n\n"
        "- best_mobilenetv2_8class.keras\n"
        "- skinlesion_8class_labels.json"
    )
    st.stop()

try:
    model = load_model()
    labels = load_labels()

    # Support either a list of labels or a JSON dictionary.
    if isinstance(labels, dict):
        try:
            labels = [labels[str(i)] for i in range(len(labels))]
        except KeyError:
            labels = list(labels.values())

    if len(labels) != model.output_shape[-1]:
        st.error(
            f"Label/model mismatch: {len(labels)} labels but "
            f"{model.output_shape[-1]} model outputs."
        )
        st.stop()

except Exception as e:
    st.error(f"Could not load the model or labels: {e}")
    st.stop()


# -------------------- MAIN WORKSPACE --------------------
left, right = st.columns([1.05, 0.95], gap="large")

with left:
    st.markdown(
        """
        <div class="section-title">01 / Upload image</div>
        <div class="section-desc">
            Choose a clear JPG, JPEG or PNG image to begin.
        </div>
        <div class="panel">
            <div class="panel-heading">Image workspace</div>
            <div class="panel-sub">
                For this research demo, use an image that resembles the
                type of skin images used to train the model.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Drop your image here or browse files",
        type=["jpg", "jpeg", "png"],
        help="Image processing happens in this Streamlit app.",
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Selected image", width="stretch")

            st.caption(
                f"Image size: {image.width} × {image.height} pixels"
            )

            analyze = st.button(
                "✦  Analyze image",
                type="primary",
                use_container_width=True,
            )

        except Exception as e:
            st.error(f"Unable to open this image: {e}")
            analyze = False
            image = None
    else:
        image = None
        analyze = False
        st.info("Your image preview will appear here after you upload a file.")

with right:
    st.markdown(
        """
        <div class="section-title">02 / Prediction results</div>
        <div class="section-desc">
            Review the model output after running an analysis.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if uploaded_file is None:
        st.markdown(
            """
            <div class="panel">
                <div style="font-size:36px; margin-bottom:12px;">◉</div>
                <div class="panel-heading">Ready for analysis</div>
                <div class="panel-sub">
                    Upload an image and select <b>Analyze image</b>.
                    Your top predicted class and class scores will appear here.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    elif image is not None and analyze:
        with st.spinner("Running the trained model..."):
            resized = image.resize((IMG_SIZE, IMG_SIZE))
            arr = np.asarray(resized, dtype=np.float32)
            arr = preprocess_input(arr)
            arr = np.expand_dims(arr, axis=0)

            probabilities = model.predict(arr, verbose=0)[0]
            order = np.argsort(probabilities)[::-1]
            best_index = int(order[0])
            predicted_label = str(labels[best_index])
            confidence = float(probabilities[best_index])

        st.markdown(
            f"""
            <div class="result-box">
                <div class="result-label">Top model prediction</div>
                <div class="result-name">{predicted_label}</div>
                <div class="result-note">
                    Highest output score from the trained 8-class model.
                    This is a model classification, not a medical diagnosis.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("Top output score", f"{confidence * 100:.1f}%")
        with metric_col2:
            st.metric("Classes evaluated", len(labels))

        st.caption(
            "The output score is not the probability that this is a correct "
            "medical diagnosis."
        )

        st.markdown("#### Scores across all classes")

        score_data = {
            str(labels[int(i)]): float(probabilities[int(i)])
            for i in order
        }

        st.bar_chart(score_data, horizontal=True)

        with st.expander("View exact output scores"):
            for i in order:
                st.write(
                    f"**{labels[int(i)]}** — "
                    f"{float(probabilities[int(i)]) * 100:.2f}%"
                )

        csv_text = "class,model_output_score_percent\n" + "\n".join(
            f"{labels[int(i)]},{float(probabilities[int(i)]) * 100:.4f}"
            for i in order
        )

        st.download_button(
            "Download prediction scores (CSV)",
            data=csv_text,
            file_name="skinlesionai_prediction_scores.csv",
            mime="text/csv",
            use_container_width=True,
        )

    else:
        st.markdown(
            """
            <div class="panel">
                <div style="font-size:30px; margin-bottom:10px;">✧</div>
                <div class="panel-heading">Image selected</div>
                <div class="panel-sub">
                    Select <b>Analyze image</b> on the left to run inference
                    with the existing trained model.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# -------------------- ABOUT / FOOTER --------------------
st.divider()

with st.expander("About this project and its limitations"):
    st.write(
        """
        **Model:** MobileNetV2 transfer-learning classifier.

        **Output classes:** akiec, bcc, bkl, df, mel, normal, nv and vasc.

        **Intended use:** Educational demonstration of image classification.

        **Important limitations:** Performance may vary across image sources,
        skin tones, lighting, camera devices and clinical settings. A high
        output score does not establish diagnostic correctness. This model has
        not been validated for clinical use and must not replace professional
        assessment.
        """
    )

st.markdown(
    """
    <div class="footer">
        <b>SkinLesionAI</b> · Student machine-learning research project<br>
        Built with Streamlit + TensorFlow · MobileNetV2<br>
        For education and research only — not for medical diagnosis.
    </div>
    """,
    unsafe_allow_html=True,
)