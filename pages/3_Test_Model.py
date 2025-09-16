import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications import efficientnet, convnext
import os
import cv2
import gdown

st.cache_data.clear()
st.cache_resource.clear()

# ---------------------------
# Custom CSS
# ---------------------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: linear-gradient(135deg, #1e1e2f 0%, #2c2c54 100%);
  background-attachment: fixed; min-height: 100vh; font-family: 'Segoe UI', sans-serif; color: #f5f5f5; }
[data-testid="stHeader"] { background: rgba(0,0,0,0); }
.app-title { position: fixed; top: 15px; left: 25px; font-size: 26px; font-weight: bold;
  color: #2c3e50; background-color: rgba(255, 255, 255, 0.85); padding: 10px 18px;
  border-radius: 12px; z-index: 9999; user-select: none; box-shadow: 0px 4px 12px rgba(0,0,0,0.1); }
[data-testid="stAppViewContainer"] > .main { background-color: rgba(255, 255, 255, 0.9);
  border-radius: 20px; padding: 50px 70px; max-width: 1000px; margin: 100px auto 60px auto;
  color: #2c3e50; backdrop-filter: blur(6px); box-shadow: 0px 8px 30px rgba(0,0,0,0.12); }
h1,h2,h3 { color: #2c3e50; }
</style>
<div class="app-title">🌿 Crop Disease App</div>
""", unsafe_allow_html=True)

# ---------------------------
# App Title
# ---------------------------
st.title("Multiple Crop Disease Detector App")
st.write("Hello User! This app analyzes crop leaves and predicts diseases.")

# ---------------------------
# Model Info (with Drive file IDs)
# ---------------------------
MODEL_DIR = "models_downloaded"
os.makedirs(MODEL_DIR, exist_ok=True)

crop_model = {
    "Rice": {
        "file_id": "1unF_iGaJtGnLBoO4SDDbAKHgEjSqrnxc",
        "filename": "rice_model_best.h5",
        "class_names": [
            'bacterial_leaf_blight', 'brown_spot', 'healthy', 'leaf_blast',
            'leaf_scald', 'narrow_brown_spot', 'neck_blast',
            'rice_hispa', 'sheath_blight', 'tungro'
        ],
        "preprocess": efficientnet.preprocess_input
    },
    "Cotton": {
        "file_id": "12IRF-aoR8-VFULzlZT0FvYtWts0jwbcE",
        "filename": "Cotton_Efficientnetb3.keras",
        "class_names": [
            'Bacterial Blight', 'Curl Virus', 'Healthy Leaf',
            'Herbicide Growth Damage', 'Leaf Hopper Jassids',
            'Leaf Redding', 'Leaf Variegation'
        ],
        "preprocess": efficientnet.preprocess_input
    },
    "Sugarcane": {
        "file_id": "1S8YKxAaimfNZc5pBihuWXlbeCNvpmbS_",
        "filename": "convnext_tiny_best.keras",
        "class_names": [
            'Banded Chlorosis', 'Brown Spot', 'BrownRust', 'Dried Leaves',
            'Grassy shoot', 'Healthy Leaves', 'Pokkah Boeng', 'Sett Rot',
            'Viral Disease', 'Yellow Leaf', 'smut'
        ],
        "preprocess": convnext.preprocess_input
    }
}

# ---------------------------
# Cached loader from Drive
# ---------------------------
@st.cache_resource
def load_model_from_drive(file_id, filename):
    local_path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(local_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, local_path, quiet=False)
    return tf.keras.models.load_model(local_path, compile=False)

# ---------------------------
# Helper: find last conv layer name
# ---------------------------
def find_last_conv_layer(model):
    common_names = ["top_conv", "conv_pw"]
    for name in common_names:
        try:
            _ = model.get_layer(name)
            return name
        except Exception:
            pass
    for layer in reversed(model.layers):
        try:
            out_shape = layer.output_shape
        except Exception:
            continue
        if not out_shape:
            continue
        if isinstance(out_shape, (list, tuple)) and len(out_shape) >= 4:
            lname = layer.name.lower()
            cls = layer.__class__.__name__.lower()
            if "conv" in lname or "conv" in cls or "depthwise" in cls:
                return layer.name
    return None

# ---------------------------
# Grad-CAM Function
# ---------------------------
def generate_gradcam(model, img_pil, preprocess_fn, last_conv_layer_name=None, alpha=0.4):
    try:
        input_size = model.input_shape[1:3]
        if input_size is None or len(input_size) != 2:
            st.warning("Model has unexpected input shape; cannot generate Grad-CAM.")
            return None, None, None

        img_resized = img_pil.resize(input_size, Image.BILINEAR)
        if img_resized.mode != "RGB":
            img_resized = img_resized.convert("RGB")

        img_array = np.array(img_resized, dtype=np.float32)
        if img_array.ndim == 2:
            img_array = np.stack((img_array,) * 3, axis=-1)
        elif img_array.shape[-1] != 3:
            img_array = img_array[..., :3]

        img_batch = np.expand_dims(img_array, axis=0)
        img_pp = preprocess_fn(img_batch.copy())

        preds = model.predict(img_pp, verbose=0)
        pred_idx = int(np.argmax(preds[0]))
        confidence = float(preds[0][pred_idx])

        if last_conv_layer_name is None:
            last_conv_layer_name = find_last_conv_layer(model)
        if last_conv_layer_name is None:
            st.info("Could not find a convolutional layer for Grad-CAM.")
            return None, confidence, preds[0]

        grad_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_pp)
            loss = predictions[:, pred_idx]

        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            st.info("Gradients are None — Grad-CAM not available for this model.")
            return None, confidence, preds[0]

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
        conv_outputs = conv_outputs[0].numpy()

        for i in range(pooled_grads.shape[-1]):
            conv_outputs[..., i] *= pooled_grads[i]

        heatmap = np.sum(conv_outputs, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1e-10

        img_cv = np.array(img_resized).astype(np.uint8)
        heatmap_resized = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        superimposed = cv2.addWeighted(img_cv, 1 - alpha, heatmap_rgb, alpha, 0)
        superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)

        return superimposed, confidence, preds[0]
    except Exception as ex:
        st.warning(f"Grad-CAM generation failed: {ex}")
        return None, None, None

# ---------------------------
# Prediction Function
# ---------------------------
def predict_disease(model, preprocess_fn, classnames, img_pil, crop_name):
    # Set input size according to model
    if crop_name in ["Rice", "Cotton"]:
        input_size = (224, 224)
    else:
        input_size = model.input_shape[1:3]

    # Force RGB
    if img_pil.mode != "RGB":
        img_pil = img_pil.convert("RGB")

    # Resize
    img_pil = img_pil.resize(input_size, Image.BILINEAR)
    img_array = np.array(img_pil, dtype=np.float32)

    # Ensure 3 channels
    if img_array.ndim == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)
    elif img_array.shape[-1] != 3:
        img_array = img_array[..., :3]

    img_batch = np.expand_dims(img_array, axis=0)
    img_batch = preprocess_fn(img_batch)

    preds = model.predict(img_batch, verbose=0)
    top_idx = int(np.argmax(preds[0]))
    confidence = float(preds[0][top_idx])
    pred_class = classnames[top_idx]
    top3_idx = np.argsort(preds[0])[-3:][::-1]
    top3 = [(classnames[i], preds[0][i]) for i in top3_idx]
    return pred_class, confidence, top3, preds[0]

# ---------------------------
# Streamlit Interface
# ---------------------------
selected = st.selectbox("Choose the crop you want to analyze:", list(crop_model.keys()))
uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg","png","jpeg"])
show_gradcam = st.checkbox("Show Grad-CAM visualization", value=True)

if uploaded_file is not None:
    try:
        img_pil = Image.open(uploaded_file).convert("RGB")
        model = load_model_from_drive(crop_model[selected]["file_id"], crop_model[selected]["filename"])
        classnames = crop_model[selected]["class_names"]
        preprocess_fn = crop_model[selected]["preprocess"]

        with st.spinner("Analyzing image..."):
            pred_class, confidence, top3, raw_preds = predict_disease(model, preprocess_fn, classnames, img_pil, selected)

        col1, col2 = st.columns([1,1])
        with col1:
            st.image(img_pil, caption="Processed Image", use_container_width=True)
            st.success(f"**Prediction:** {pred_class} ({confidence*100:.2f}%)")
            st.subheader("Top 3 Predictions:")
            for name, conf in top3:
                st.write(f"{name}: {conf*100:.2f}%")

        if show_gradcam:
            with st.spinner("Generating Grad-CAM..."):
                last_conv = None
                if selected == "Sugarcane":
                    last_conv = "convnext_tiny_stage_3_block_2_pointwise_conv_2"
                gradcam_img, g_conf, preds = generate_gradcam(model, img_pil, preprocess_fn, last_conv_layer_name=last_conv)

            with col2:
                if gradcam_img is not None:
                    st.subheader("Grad-CAM Visualization")
                    st.image(gradcam_img, caption=f"Grad-CAM: {pred_class} ({(g_conf*100):.2f}%)", use_container_width=True)
                else:
                    st.info("Grad-CAM visualization not available for this model or failed to generate.")

    except Exception as e:
        st.error("Error during model prediction. See details below:")
        st.exception(e)
