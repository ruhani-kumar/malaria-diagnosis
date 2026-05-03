import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/malaria_finetuned_model.keras")
model = load_model()

def monte_carlo_prediction(model, img_array, n_iter=30):
    preds = []
    for _ in range(n_iter):
        noisy = img_array + np.random.normal(0, 0.01, img_array.shape)
        pred = model(img_array, training=True).numpy()[0][0]
        preds.append(pred)
    preds = np.array(preds)
    mean = preds.mean()
    std = preds.std()
    return mean, std

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image

st.title("Malaria Detection System")

st.header("Model Comparison")

st.table({
    "Model": ["LeNet", "MobileNetV2", "Fine-tuned MobileNetV2"],
    "Accuracy": ["99.56%", "95.51%", "99.78%"],
    "Validation Accuracy": ["94.59%", "95%", "96.5%"],
    "Test Accuracy": ["93.73%", "94.56%", "95.72%"]
})
st.divider()
st.header("Training Curves")
st.subheader("Loss Comparison")

col1, col2, col3 = st.columns(3)
with col1:
    st.image("assets/loss_curves/lenet_loss.png", caption="LeNet Loss")
with col2:
    st.image("assets/loss_curves/mobilenet_loss.png", caption="MobileNetV2 Loss")
with col3:
    st.image("assets/loss_curves/mobilenetFinetune_loss.png", caption="Fine-tuned Loss")

st.subheader("Accuracy Comparison")
col1, col2, col3 = st.columns(3)
with col1:
    st.image("assets/accuracy_curves/lenet_acc.png", caption="LeNet Accuracy")
with col2:
    st.image("assets/accuracy_curves/mobilenet_acc.png", caption="MobileNetV2 Accuracy")
with col3:
    st.image("assets/accuracy_curves/mobilenetFinetune_acc.png", caption="Fine-tuned Accuracy")
st.divider()

st.header("Monte Carlo Uncertainity")
col1, col2 = st.columns(2)
with col1:
    st.image("assets/montecarlo/mc_image_1.png", caption="Input Image")
with col2:
    st.image("assets/montecarlo/mc_hist_1.png", caption="Prediction Distribution")

st.header("Grad CAM Explainability")
col1, col2 = st.columns(2)
with col1:
    st.image("assets/gradcam/sample_2_Parasitized.jpg", caption="Input Image")
with col2:
    st.image("assets/gradcam/gradcam_2.jpg", caption="Heatmap Explanation")
st.divider()
col1, col2 = st.columns(2)
with col1:
    st.image("assets/gradcam/sample_0_Uninfected.jpg", caption="Input Image")
with col2:
    st.image("assets/gradcam/gradcam_0.jpg", caption="Heatmap Explanation")

st.header("Upload Image for Prediction")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_array = preprocess_image(image)
    pred = model.predict(img_array, verbose=0)[0][0]
    mean_pred, uncertainty = monte_carlo_prediction(model, img_array)
    if pred < 0.5:
        label = "Parasitized"
        confidence = mean_pred
    else:
        label = "Uninfected"
        confidence = 1 - mean_pred
    st.subheader("Prediction Result")
    st.success(f"Prediction: {label}")
    st.info(f"Confidence: {confidence:.4f}")