import cv2
import numpy as np
from scipy.spatial.distance import euclidean

def compute_hog(image):
    """Compute the HoG descriptor for a given image."""
    hog_descriptor = cv2.HOGDescriptor()
    hog_features = hog_descriptor.compute(image)
    return hog_features.flatten()

def compute_reference_hogs(reference_images):
    """Compute HoG descriptors for reference images."""
    hog_descriptors = []
    hog_descriptor = cv2.HOGDescriptor()
    
    for img in reference_images:
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hog_features = hog_descriptor.compute(gray_image)
        hog_descriptors.append(hog_features.flatten())
    
    return hog_descriptors

def extract_hog_features(image, window_size=(64, 128), step_size=16):
    """Extract HoG features from sliding windows of an image."""
    hog_descriptor = cv2.HOGDescriptor()
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray_image.shape
    
    windows = []
    hog_descriptors = []
    for y in range(0, height - window_size[1], step_size):
        for x in range(0, width - window_size[0], step_size):
            window = gray_image[y:y+window_size[1], x:x+window_size[0]]
            window = cv2.resize(window, (window_size[0], window_size[1]))  # Ensure window size consistency
            hog_features = hog_descriptor.compute(window).flatten()
            windows.append((x, y, window_size))
            hog_descriptors.append(hog_features)
    
    return windows, hog_descriptors

def calculate_similarity(hog_features_window, hog_descriptors_ref):
    """Calculate similarity scores between window HoG and reference HoG descriptors."""
    similarities = []
    for ref_descriptor in hog_descriptors_ref:
        if hog_features_window.shape == ref_descriptor.shape:
            distance = euclidean(hog_features_window, ref_descriptor)
            similarities.append(distance)
        else:
            print(f"Shape mismatch: {hog_features_window.shape} vs {ref_descriptor.shape}")
    
    return similarities

def detect_humans(image, reference_hogs, threshold=0.5, window_size=(64, 128), step_size=16):
    """Detect humans in an image using HoG descriptor and sliding window approach."""
    windows, hog_descriptors = extract_hog_features(image, window_size, step_size)
    detected_boxes = []

    for (x, y, window_size), hog_features_window in zip(windows, hog_descriptors):
        similarities = calculate_similarity(hog_features_window, reference_hogs)
        if len(similarities) > 0 and min(similarities) < threshold:
            detected_boxes.append((x, y, window_size))
    
    detected_boxes = non_max_suppression(detected_boxes)
    
    return detected_boxes

def non_max_suppression(boxes, overlap_thresh=0.3):
    """Apply non-maximum suppression to remove overlapping bounding boxes."""
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2][0]
    y2 = boxes[:, 1] + boxes[:, 2][1]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))

    return [boxes[i] for i in pick]

# Example usage
if __name__ == "__main__":
    # Load reference images
    reference_images = [cv2.imread("/home/student/PycharmProjects/220962076_CV/220962076_week7/images2.jpeg"), cv2.imread("girl-1894125_640.jpg")]
    reference_hogs = compute_reference_hogs(reference_images)
    
    # Load input image
    input_image = cv2.imread("human3.jpeg")
    
    # Detect humans
    detected_boxes = detect_humans(input_image, reference_hogs, threshold=0.5)
    
    # Draw bounding boxes on the image
    for (x, y, (w, h)) in detected_boxes:
        cv2.rectangle(input_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Handle display errors for headless environments
    try:
        cv2.imshow("Detected Humans", input_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error as e:
        print("OpenCV display error:", e)
        # Save the result instead
        cv2.imwrite("detected_humans.jpg", input_image)
        print("Detected humans image saved as 'detected_humans.jpg'")
