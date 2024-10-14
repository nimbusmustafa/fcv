import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

def correct_skew(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated, angle

def draw_rectangle(image, angle):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    rect = cv2.minAreaRect(coords)
    box = cv2.boxPoints(rect)
    box = np.int32(box)  # Use np.int32 instead of np.int0
    image_with_rect = cv2.drawContours(image.copy(), [box], 0, (0, 255, 0), 2)
    return image_with_rect

st.title('Text Skew Correction with Streamlit')

uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    image_bytes = uploaded_file.read()
    image = Image.open(io.BytesIO(image_bytes))
    image = np.array(image)

    corrected_image, original_angle = correct_skew(image)
    corrected_image_pil = Image.fromarray(corrected_image)
    image_with_rect = draw_rectangle(image, original_angle)
    image_with_rect_pil = Image.fromarray(image_with_rect)

    st.image(image_with_rect_pil, caption=f'Original Image with Skew Angle: {original_angle:.2f} degrees', use_column_width=True)
    st.image(corrected_image_pil, caption='Corrected Image', use_column_width=True)

    buf = io.BytesIO()
    corrected_image_pil.save(buf, format='PNG')
    byte_im = buf.getvalue()
    st.download_button(label="Download Corrected Image", data=byte_im, file_name='corrected_image.png', mime='image/png')
