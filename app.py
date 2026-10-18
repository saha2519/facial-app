import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

BASE = Path(__file__).resolve().parent
st.set_page_config(page_title="CapsNet Facial Expression Recognition", page_icon="🙂")
st.title("Facial Expression Recognition using Capsule Networks")
st.caption("FER-2013 · 28 × 28 grayscale · Seven expression categories")

@st.cache_resource
def load_inference():
    with (BASE / "classes.json").open(encoding="utf-8") as f:
        labels = json.load(f)
    loaded = tf.saved_model.load(str(BASE / "saved_model"))
    fn = getattr(loaded, "serve", None)
    if fn is None:
        fn = loaded.signatures["serving_default"]
    return labels, fn

try:
    class_names, inference_fn = load_inference()
except Exception as exc:
    st.error(f"Unable to load model files: {exc}")
    st.stop()

uploaded = st.file_uploader("Upload a cropped face image", type=["png", "jpg", "jpeg"])
if uploaded is not None:
    image = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")
    st.image(image, caption="Uploaded image", width=280)
    if st.button("Predict expression", type="primary"):
        small = image.convert("L").resize((28, 28))
        arr = np.asarray(small, dtype=np.float32)[None, :, :, None] / 255.0
        result = inference_fn(tf.constant(arr))
        if isinstance(result, dict):
            result = next(iter(result.values()))
        scores = np.asarray(result)[0]
        if len(scores) != len(class_names):
            st.error("Model output and class labels do not match.")
            st.stop()
        index = int(np.argmax(scores))
        st.subheader(f"Predicted expression: {class_names[index].title()}")
        st.write("Capsule scores (not calibrated probabilities):")
        st.bar_chart({name: float(value) for name, value in zip(class_names, scores)})

st.info("The model classifies visible facial expressions, not a person's actual internal emotions. Predictions may be inaccurate.")
