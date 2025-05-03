import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the model once
model = tf.keras.models.load_model("cifar10_image_classifier.h5")

# CIFAR-10 class labels
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Image preprocessing function
def preprocess_image(image):
    image = image.resize((32, 32))
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Company branding (top-left)
st.markdown(
    """
    <div style="display: flex; align-items: center;">
        <img src="https://integrano.com/wp-content/uploads/2019/10/integrano-logo.jpg" width="150" style="margin-right: 15px;">
        <h1 style="margin: 0; font-size: 24px; font-weight: 600">Integrano Technologies Pvt. Ltd.</h1>
    </div>
    """,
    unsafe_allow_html=True
)
# Streamlit interface
st.title("🔍 CIFAR-10 Image Classifier")
st.write("Upload an image and the AI will tell you what it is.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Predict
    if st.button("Classify"):
        processed = preprocess_image(image)
        predictions = model.predict(processed)
        predicted_class = class_names[np.argmax(predictions)]
        confidence = 100 * np.max(tf.nn.softmax(predictions[0]))

        st.success(f"Prediction: **{predicted_class}** ({confidence:.2f}% confidence)")
