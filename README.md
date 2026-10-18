# CapsNet Facial Expression Recognition

Streamlit app for the TensorFlow SavedModel exported from the trained Capsule Network.

Before deployment, add `classes.json` and extract `capsnet_saved_model.zip` so that the `saved_model/` directory sits alongside `app.py`.

Deploy on Streamlit Community Cloud with main file `app.py`.

Note: TensorFlow may exceed memory limits of free hosting tiers. If that happens, use a hosting tier with enough RAM or run locally.
