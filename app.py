import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

IMG_SIZE = (224, 224)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("chest_xray_final.keras")

tf.config.set_visible_devices([], "GPU")

model = load_model()


def preprocess_image(image_file):
    img = load_img(image_file, target_size=IMG_SIZE, color_mode="rgb")
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

st.title("Chest X-ray Pneumonia Detection")
# st.warning(
#     "This tool is for educational purposes and is not for medical diagnosis yet."
# )

threshold = st.slider(
    "Decision threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05
)


st.write("Upload a chest X-ray image to assess pneumonia risk.")

uploaded_file = st.file_uploader("Upload X-ray", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded X-ray", use_column_width=True)

    img = preprocess_image(uploaded_file)
    
    with st.spinner("Analyzing X-ray..."):
        prob = model.predict(img)[0][0]

    st.subheader("Prediction")
    st.write(f"Pneumonia probability: **{prob:.3f}**")
    st.write(f"Confidence level: **{abs(prob - 0.5) * 2:.2f}**")


    if prob >= threshold:
        st.error("Pneumonia detected")
    else:
        st.success("Normal")

   

st.markdown("---")
st.caption("This application is for educational and research purposes only.")
