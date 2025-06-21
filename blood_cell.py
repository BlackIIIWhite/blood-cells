import streamlit as st
import tensorflow as tf
import numpy as np

# Load the model
try:
    model = tf.keras.models.load_model('model_compat.h5')
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Class names
class_name = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']

def prediction(images):
    # Predict the class index
    prediction_idx = np.argmax(model.predict(images))
    # Debug to Streamlit console
    st.write("Predicted Index:", prediction_idx)
    return class_name[prediction_idx]

# App Title
st.title("Blood Cell Type Prediction")

# Upload Image
uploaded_file = st.file_uploader("Choose a blood cell image (JPG)...", type="jpg", key="image")

if uploaded_file is not None:
    # Load and preprocess image
    image = tf.keras.preprocessing.image.load_img(uploaded_file, target_size=(256, 256))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Predict
    predicted_class = prediction(image)
    st.write("Predicted Cell Type:", predicted_class)

else:
    st.info("Please upload an image file.")
