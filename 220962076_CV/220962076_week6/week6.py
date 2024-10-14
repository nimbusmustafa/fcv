import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

def resize_mask(image, mask):
    return cv2.resize(mask, (image.shape[1], image.shape[0]))

def inpaint_image(image, mask):
    return cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

def convert_to_8bit_color(image):
    if len(image.shape) == 2:  # If grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:  # If image has alpha channel
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image

def convert_to_8bit_binary(mask):
    mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
    return mask

st.title('Image Inpainting with Streamlit')

uploaded_image = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])
uploaded_mask = st.file_uploader("Upload Mask", type=['png', 'jpg', 'jpeg'])

if uploaded_image and uploaded_mask:
    image_bytes = uploaded_image.read()
    image = Image.open(io.BytesIO(image_bytes))
    image = np.array(image)

    image = convert_to_8bit_color(image)

    mask_bytes = uploaded_mask.read()
    mask = Image.open(io.BytesIO(mask_bytes)).convert('L')  # Convert to grayscale
    mask = np.array(mask)

    mask = convert_to_8bit_binary(mask)

    if image.shape[:2] != mask.shape[:2]:
        mask = resize_mask(image, mask)

    inpainted_image = inpaint_image(image, mask)

    inpainted_image_pil = Image.fromarray(inpainted_image)

    st.image(inpainted_image_pil, caption='Inpainted Image', use_column_width=True)
    st.download_button("Download Inpainted Image", data=cv2.imencode('.jpg', inpainted_image)[1].tobytes(), file_name='inpainted_image.jpg')
