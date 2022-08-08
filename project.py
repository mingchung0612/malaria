#Image Processing
import keras
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import streamlit as st
import cv2
import matplotlib.pyplot as plt
import time
import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries

model = tf.keras.models.load_model("Malaria.h5")

def main():
    st.title("Image Classification")
    st.text("Upload a Malaria Image to check it is uninfected or parasite")
    class_names = ['Uninfected',"Parasite"]
    upload_image = st.file_uploader("Choose File",type=['png','jpg','jpeg'])
    submit = st.button("Show Result")
    if upload_image is not None:
        image = Image.open(upload_image)
        st.image(image,caption='Upload Image',use_column_width=True)

    if submit:
        if upload_image is None:
            st.error("Invalid Input,Please upload again the image.")
        else:
            with st.spinner("Classifying..."):
                plt.imshow(image)
                plt.axis("On")
                time.sleep(2)
                predictions = model.predict([prepare(upload_image.name)])
                convert = predictions / 255.0
                con = np.argmax(convert)              
                st.success("The image is " + class_names[con])

                st.subheader("LIME EXPLAINER")
                explainer = lime_image.LimeImageExplainer()
                explanation = explainer.explain_instance(explain_prepare(upload_image.name),model.predict, top_labels=3, hide_color=0)

                temp_1, mask_1 = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=15, hide_rest=True)
                temp_2, mask_2 = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=15, hide_rest=False)
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,15))
                ax1.imshow(mark_boundaries(temp_1, mask_1))
                ax2.imshow(mark_boundaries(temp_2, mask_2))
                ax1.axis('off')
                ax2.axis('off')
                st.pyplot(fig)

def prepare(img):
    IMG_SIZE =  224 # 224 based on the size you train for model
    img_array = cv2.imread(img)  # read in the image
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
    return new_array.reshape(-1,IMG_SIZE, IMG_SIZE, 3)

def explain_prepare(images):
    IMG_SIZE = 224  # 180 based on the size you train for model
    img_array = cv2.imread(images)  # read in the image
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array

if __name__ == "__main__":
    main()