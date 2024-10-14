import cv2
import numpy as np
import matplotlib.pyplot as plt


def global_thresholding(image):
    """
    Apply global thresholding to convert an image to binary.
    """
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    return binary_image


def adaptive_thresholding(image):
    """
    Apply adaptive thresholding to convert an image to binary.
    """
    binary_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return binary_image


def detect_lines(image_path):
    """
    Detect lines in an image using Hough Transform.
    """
    # Read the image in grayscale
    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply edge detection
    edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)

    # Detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)

    # Read the original image in color
    color_image = cv2.imread(image_path)

    # Draw lines on the image
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            cv2.line(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)


def color_segmentation(image, lower_bound, upper_bound):
    """
    Segment an image based on color using HSV color space.
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, np.array(lower_bound), np.array(upper_bound))
    segmented_image = cv2.bitwise_and(image, image, mask=mask)
    return cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)


def k_means_clustering(image, k=3):
    """
    Apply K-Means clustering to segment an image into k clusters.
    """
    Z = image.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)
    return cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)


def canny_edge_detection(image):
    """
    Perform Canny Edge Detection on an image.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 1.5)

    sobel_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    angle = np.arctan2(sobel_y, sobel_x) * 180 / np.pi
    angle[angle < 0] += 180

    magnitude = np.uint8(255 * magnitude / np.max(magnitude))

    # Non-maximum suppression
    non_max_supp = np.zeros_like(magnitude)
    for y in range(1, magnitude.shape[0] - 1):
        for x in range(1, magnitude.shape[1] - 1):
            grad_angle = angle[y, x]
            if (0 <= grad_angle < 22.5) or (157.5 <= grad_angle <= 180):
                q, r = magnitude[y, x + 1], magnitude[y, x - 1]
            elif (22.5 <= grad_angle < 67.5):
                q, r = magnitude[y + 1, x - 1], magnitude[y - 1, x + 1]
            elif (67.5 <= grad_angle < 112.5):
                q, r = magnitude[y + 1, x], magnitude[y - 1, x]
            elif (112.5 <= grad_angle < 157.5):
                q, r = magnitude[y - 1, x - 1], magnitude[y + 1, x + 1]
            if magnitude[y, x] >= q and magnitude[y, x] >= r:
                non_max_supp[y, x] = magnitude[y, x]

    # Double thresholding
    high_threshold = np.percentile(non_max_supp, 90)
    low_threshold = 0.5 * high_threshold
    edges = np.zeros_like(non_max_supp)
    strong = (non_max_supp >= high_threshold)
    weak = (non_max_supp >= low_threshold) & (non_max_supp < high_threshold)
    edges[strong] = 255
    edges[weak] = 75

    # Edge tracking by hysteresis
    for y in range(1, edges.shape[0] - 1):
        for x in range(1, edges.shape[1] - 1):
            if edges[y, x] == 75:
                if ((edges[y + 1, x] == 255) or (edges[y - 1, x] == 255) or
                        (edges[y, x + 1] == 255) or (edges[y, x - 1] == 255) or
                        (edges[y + 1, x + 1] == 255) or (edges[y - 1, x - 1] == 255) or
                        (edges[y + 1, x - 1] == 255) or (edges[y - 1, x + 1] == 255)):
                    edges[y, x] = 255
                else:
                    edges[y, x] = 0

    return cv2.cvtColor(np.uint8(edges), cv2.COLOR_GRAY2RGB)


def main():
    # Path to the image file
    image_path = 'sample.jpg'
    image = cv2.imread(image_path)

    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    global_thresh = global_thresholding(gray_image)
    adaptive_thresh = adaptive_thresholding(gray_image)

    # Detect lines in the image
    lines_image = detect_lines(image_path)

    # Apply color segmentation
    lower_bound = [0, 100, 100]
    upper_bound = [10, 255, 255]
    color_segmented = color_segmentation(image, lower_bound, upper_bound)

    # Apply K-Means clustering
    kmeans_img = k_means_clustering(image, k=3)

    # Apply Canny edge detection
    canny_edges = canny_edge_detection(image)

    # Display results
    plt.figure(figsize=(12, 12))

    plt.subplot(3, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(3, 3, 2)
    plt.imshow(global_thresh, cmap='gray')
    plt.title('Global Thresholding')
    plt.axis('off')

    plt.subplot(3, 3, 3)
    plt.imshow(adaptive_thresh, cmap='gray')
    plt.title('Adaptive Thresholding')
    plt.axis('off')

    plt.subplot(3, 3, 4)
    plt.imshow(lines_image)
    plt.title('Hough Line Detection')
    plt.axis('off')

    plt.subplot(3, 3, 5)
    plt.imshow(color_segmented)
    plt.title('Color Segmentation')
    plt.axis('off')

    plt.subplot(3, 3, 6)
    plt.imshow(kmeans_img)
    plt.title('K-Means Clustering')
    plt.axis('off')

    plt.subplot(3, 3, 7)
    plt.imshow(canny_edges)
    plt.title('Canny Edge Detection')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
