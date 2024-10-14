import cv2
import numpy as np
import streamlit as st
from PIL import Image

def harris_corner_detection(image, k=0.04, threshold=0.01):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    Ix = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    
    Ixx = Ix**2
    Ixy = Ix*Iy
    Iyy = Iy**2
    
    Sxx = cv2.GaussianBlur(Ixx, (3, 3), 0)
    Sxy = cv2.GaussianBlur(Ixy, (3, 3), 0)
    Syy = cv2.GaussianBlur(Iyy, (3, 3), 0)
    
    det_M = Sxx * Syy - Sxy**2
    trace_M = Sxx + Syy
    R = det_M - k * trace_M**2
    
    corners = R > (threshold * R.max())
    corners_image = image.copy()
    corners_image[corners] = [0, 0, 255]  # Mark corners in red
    
    return corners_image

def fast_corner_detection(image, threshold):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    corners_image = image.copy()
    height, width = gray.shape
    
    # Define the circle of 16 pixels around the center pixel
    circle = [(int(3*np.sin(i*2*np.pi/16)), int(3*np.cos(i*2*np.pi/16))) for i in range(16)]
    
    for y in range(3, height-3):
        for x in range(3, width-3):
            center_intensity = gray[y, x]
            circle_intensities = [gray[y+dy, x+dx] for dx, dy in circle]
            
            # Compute the differences in intensity and check the FAST corner criterion
            differences = np.abs(np.array(circle_intensities) - center_intensity)
            if np.sum(differences > threshold) > 12:
                corners_image[y, x] = [0, 255, 0]  # Mark corners in green
    
    return corners_image

# Streamlit app
st.title('Corner Detection Visualization')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)
    image = np.array(image)
    
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Add sliders for thresholds
    harris_threshold = st.slider('Harris Corner Detection Threshold', 0.0, 0.5, 0.01)
    fast_threshold = st.slider('FAST Corner Detection Threshold', 0, 100, 10)

    # Harris Corner Detection
    harris_image = harris_corner_detection(image.copy(), threshold=harris_threshold)
    st.subheader('Harris Corner Detection')
    st.image(harris_image, caption='Harris Corners', use_column_width=True)

    # FAST Corner Detection
    fast_image = fast_corner_detection(image.copy(), threshold=fast_threshold)
    st.subheader('FAST Corner Detection')
    st.image(fast_image, caption='FAST Corners', use_column_width=True)