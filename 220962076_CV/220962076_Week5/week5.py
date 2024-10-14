import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Red color HSV bounds
RED_COLOR_BOUNDS = ([0, 50, 50], [10, 255, 255]), ([160, 50, 50], [180, 255, 255])

def detect_lines_with_hough(image, threshold, min_line_length, max_line_gap):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)

    color_image = image.copy()
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            cv2.line(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

def detect_edges_with_canny(image, low_threshold, high_threshold, aperture_size):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold, apertureSize=aperture_size)

    color_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return color_image

def segment_red_color(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for red color in HSV
    lower_bound1, upper_bound1 = RED_COLOR_BOUNDS[0]
    lower_bound2, upper_bound2 = RED_COLOR_BOUNDS[1]
    lower_bound1 = np.array(lower_bound1)
    upper_bound1 = np.array(upper_bound1)
    lower_bound2 = np.array(lower_bound2)
    upper_bound2 = np.array(upper_bound2)

    # Create masks for red color
    mask1 = cv2.inRange(hsv_image, lower_bound1, upper_bound1)
    mask2 = cv2.inRange(hsv_image, lower_bound2, upper_bound2)
    mask = mask1 | mask2

    # Apply the mask to the image
    red_segmented = cv2.bitwise_and(image, image, mask=mask)

    return cv2.cvtColor(red_segmented, cv2.COLOR_BGR2RGB)

def main():
    st.set_page_config(page_title="Image Processing Tool", layout="wide")

    st.title("Image Processing Tool")
    st.markdown(
        """
        This tool allows you to apply Canny edge detection, Hough line detection, or red color segmentation to your images.
        Choose the desired processing method from the sidebar.
        """
    )

    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image)

        st.sidebar.header("Select Processing Method")
        method = st.sidebar.radio("Choose Processing Method",
                                  ("Canny Edge Detection", "Hough Line Detection", "Red Color Segmentation"))

        if method == "Canny Edge Detection":
            st.sidebar.header("Canny Edge Parameters")
            low_threshold = st.sidebar.slider('Low Threshold', min_value=0, max_value=100, value=50, step=1)
            high_threshold = st.sidebar.slider('High Threshold', min_value=0, max_value=300, value=150, step=1)
            aperture_size = st.sidebar.selectbox('Aperture Size', [3, 5, 7], index=0)

            result_image = detect_edges_with_canny(image, low_threshold, high_threshold, aperture_size)
            st.image(result_image, caption='Detected Edges', use_column_width=True)

        elif method == "Hough Line Detection":
            st.sidebar.header("Hough Transform Parameters")
            threshold = st.sidebar.slider('Threshold', min_value=50, max_value=400, value=100, step=1)
            min_line_length = st.sidebar.slider('Min Line Length', min_value=10, max_value=400, value=50, step=1)
            max_line_gap = st.sidebar.slider('Max Line Gap', min_value=1, max_value=200, value=10, step=1)

            result_image = detect_lines_with_hough(image, threshold, min_line_length, max_line_gap)
            st.image(result_image, caption='Detected Lines', use_column_width=True)

        elif method == "Red Color Segmentation":
            result_image = segment_red_color(image)
            st.image(result_image, caption='Red Color Segmentation', use_column_width=True)

if __name__ == "__main__":
    main()
