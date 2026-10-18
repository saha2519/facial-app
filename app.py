import json
import os

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras import Model, layers

st.set_page_config(page_title="Facial Emotion Recognition | CapsNet", page_icon="🧠")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def squash(vectors, axis=-1):
    squared_norm = tf.reduce_sum(tf.square(vectors), axis=axis, keepdims=True)
    scale = squared_norm / (1.0 + squared_norm)
    return scale * vectors / tf.sqrt(squared_norm + 1e-7)


class RoutingCapsules(layers.Layer):
    def __init__(self, num_capsules=7, dim_capsule=16, routing_iterations=3, **kwargs):
        super().__init__(**kwargs)
        self.num_capsules = num_capsules
        self.dim_capsule = dim_capsule
        self.routing_iterations = routing_iterations

    def build(self, input_shape):
        self.num_input_capsules = int(input_shape[1])
        self.input_dim = int(input_shape[2])
        self.W = self.add_weight(
            name="transform_weights",
            shape=(self.num_input_capsules, self.num_capsules, self.input_dim, self.dim_capsule),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        predictions = tf.einsum("bid,ijdo->bijo", inputs, self.W)
        logits = tf.zeros(tf.shape(predictions)[:-1], dtype=predictions.dtype)
        for iteration in range(self.routing_iterations):
            coupling = tf.nn.softmax(logits, axis=-1)
            weighted = coupling[..., tf.newaxis] * predictions
            output = squash(tf.reduce_sum(weighted, axis=1))
            if iteration < self.routing_iterations - 1:
                agreement = tf.reduce_sum(
                    predictions * output[:, tf.newaxis, :, :], axis=-1
                )
                logits = logits + agreement
        return output


class SquashLayer(layers.Layer):
    def call(self, inputs):
        return squash(inputs)


class CapsuleLengths(layers.Layer):
    def call(self, inputs):
        return tf.sqrt(tf.reduce_sum(tf.square(inputs), axis=-1) + 1e-8)


def build_capsnet(num_classes):
    inputs = layers.Input(shape=(28, 28, 1), name="face_image")
    x = layers.Conv2D(
        128, kernel_size=9, strides=1, activation="relu", padding="valid",
        name="feature_extractor",
    )(inputs)
    x = layers.Conv2D(
        64, kernel_size=9, strides=2, activation="relu", padding="valid",
        name="primary_capsule_features",
    )(x)
    x = layers.Reshape((-1, 8), name="primary_capsules")(x)
    x = SquashLayer(name="primary_squash")(x)
    x = RoutingCapsules(
        num_capsules=num_classes, dim_capsule=16, routing_iterations=3,
        name="emotion_capsules",
    )(x)
    outputs = CapsuleLengths(name="emotion_probabilities")(x)
    return Model(inputs, outputs, name="FacialEmotionCapsNet")


@st.cache_data
def load_classes():
    with open(os.path.join(BASE_DIR, "classes.json"), encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def load_model(num_classes):
    model = build_capsnet(num_classes)
    model.load_weights(os.path.join(BASE_DIR, "capsnet_emotion.weights.h5"))
    return model


st.title("Facial Emotion Recognition")
st.subheader("Using Capsule Networks (CapsNet)")
st.info(
    "An AI-powered system that analyzes facial expressions and "
    "classifies them into seven emotion categories."
)

try:
    class_names = load_classes()
    model = load_model(len(class_names))
except Exception as exc:
    st.error(f"Unable to load CapsNet model: {exc}")
    st.stop()

uploaded = st.file_uploader("Upload a facial image", type=["png", "jpg", "jpeg"])
if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)
    if st.button("Predict Emotion", type="primary", use_container_width=True):
        try:
            gray = image.convert("L").resize((28, 28))
            array = np.asarray(gray, dtype=np.float32).reshape(1, 28, 28, 1) / 255.0
            scores = model.predict(array, verbose=0)[0]
            emotion = class_names[int(np.argmax(scores))]
            st.success(f"Predicted Facial Expression: {emotion.upper()}")
            st.markdown("### Capsule Scores")
            for name, score in sorted(zip(class_names, scores), key=lambda p: -p[1]):
                st.write(f"{name.capitalize()}: {float(score):.4f}")
                st.progress(float(np.clip(score, 0.0, 1.0)))
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

st.divider()
st.markdown("### About the Project")
st.write("**Algorithm:** Capsule Networks with Dynamic Routing")
st.write("**Dataset:** FER-2013")
st.write("**Input:** 28 × 28 grayscale image")
st.write("**Categories:** Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise")
st.caption("Capsule scores are not calibrated probabilities.")
