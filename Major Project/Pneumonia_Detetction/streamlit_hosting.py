import numpy as np
import streamlit as st
import tensorflow as tf
import keras
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import pickle

#with open(r"C:\Users\ojas2\Downloads\model.pkl", 'rb') as f:
#    model_pkl = pickle.load(f)

model = load_model('C:\\Users\\ojas2\OneDrive\\Desktop\TUF\\b.tech\data sciences\\Scifor\\model2.h5')
def test_image(path):
    image = load_img(path, target_size=(150, 150))
    image = img_to_array(image)
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def main():
    st.title("Image Classification Model")

    uploaded_file = st.file_uploader("Upload an image")

    if uploaded_file is not None:
        image_data = test_image(uploaded_file)

        prediction = model.predict(image_data)

        st.write("Prediction:", prediction)

if __name__ == "__main__":
    main()
