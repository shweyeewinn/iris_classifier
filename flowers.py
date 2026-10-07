import streamlit as st
import pickle
from pathlib import Path
from PIL import Image, ImageOps

st.set_page_config(page_title="Iris Classifier", page_icon="🌸", layout="wide")

# ---------- Paths & constants ----------
APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "iris_model.pkl"
SPECIES = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]  # LabelEncoder order (alphabetical)
IMAGES = {
    "Iris-setosa": APP_DIR / "iris-setosa.jpg",
    "Iris-versicolor": APP_DIR / "iris-versicolor.jpg",
    "Iris-virginica": APP_DIR / "Iris-virginica.jpg",
}
IMG_SIZE = (800, 600)  # 4:3 crop; displayed at the card's width on every screen

# ---------- Styling ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');
html, body, .stApp, p, h1, h2, h3, h4, label, button, input {
    font-family: 'Nunito', sans-serif !important;
}
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 2.5rem; max-width: 1150px;}

.stMarkdown p.hero-title {font-size: 2.6rem !important; font-weight: 800; margin: 0; letter-spacing: -0.5px; line-height: 1.2;}
.stMarkdown p.hero-sub {color: #6B6580; font-size: 1.1rem !important; margin: .3rem 0 1.6rem;}

[data-testid="stImage"] img {border-radius: 18px;}

.pill {
    display: inline-block; background: #EDE9FE; color: #5B4BD8;
    padding: 4px 12px; border-radius: 999px;
    font-size: .8rem; font-weight: 700; letter-spacing: .5px;
}
.stMarkdown p.result-name {font-size: 2rem !important; font-weight: 800; color: #5B4BD8; margin: .4rem 0 1rem; line-height: 1.2;}

/* ---------- Tablets ---------- */
@media (max-width: 900px) {
    .block-container {padding-left: 1.5rem; padding-right: 1.5rem;}
    .stMarkdown p.hero-title {font-size: 2.2rem !important;}
}
/* ---------- Phones ---------- */
@media (max-width: 640px) {
    .block-container {padding: 3.75rem 1rem 2rem;}  /* top space clears the Streamlit toolbar */
    .stMarkdown p.hero-title {font-size: 1.8rem !important;}
    .stMarkdown p.hero-sub {font-size: 1rem !important; margin-bottom: 1rem;}
    .stMarkdown p.result-name {font-size: 1.6rem !important;}
}
</style>
""", unsafe_allow_html=True)

# ---------- Helpers ----------
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_image(path):
    img = Image.open(path).convert("RGB")
    return ImageOps.fit(img, IMG_SIZE)

def to_label(p):
    return SPECIES[int(p)] if str(p).isdigit() else str(p)

model = load_model()

# ---------- Header ----------
st.markdown('<p class="hero-title">Iris Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Tell me the flower\'s measurements and I\'ll guess its species.</p>',
            unsafe_allow_html=True)

# ---------- Layout ----------
# Two columns side by side on desktop and tablet; Streamlit stacks them
# automatically on phones (inputs first, then the result).
left, right = st.columns([1, 1], gap="large")

# ---------- Inputs (on the page, not in the sidebar, so they show on every device) ----------
with left:
    with st.container(border=True):
        st.markdown("#### Measurements")
        st.caption("Move the sliders. All values are in cm.")
        sepal_length = st.slider("Sepal length", 4.0, 8.0, 5.0, 0.1)
        sepal_width = st.slider("Sepal width", 2.0, 5.0, 3.0, 0.1)
        petal_length = st.slider("Petal length", 1.0, 7.0, 4.0, 0.1)
        petal_width = st.slider("Petal width", 0.1, 2.5, 1.0, 0.1)

# ---------- Prediction (updates live) ----------
data = [[sepal_length, sepal_width, petal_length, petal_width]]
label = to_label(model.predict(data)[0])
probs = None
if hasattr(model, "predict_proba"):
    probs = sorted(
        zip([to_label(c) for c in model.classes_], model.predict_proba(data)[0]),
        key=lambda x: x[1], reverse=True,
    )

# ---------- Result ----------
with right:
    with st.container(border=True):
        st.markdown(f'<span class="pill">PREDICTION</span>'
                    f'<p class="result-name">{label}</p>', unsafe_allow_html=True)
        img_path = IMAGES.get(label)
        if img_path and img_path.exists():
            st.image(load_image(str(img_path)), width="stretch")
        else:
            st.info(f"No image found for {label}")

        if probs:
            st.markdown("#### Confidence")
            for name, p in probs:
                st.progress(float(p), text=f"{name} — {p:.0%}")
