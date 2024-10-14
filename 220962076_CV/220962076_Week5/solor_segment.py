import cv2
import numpy as np


def segment_red_color(image_path, output_path):
    # Load the image
    image = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if image is None:
        print(f"Error: Unable to load image at path '{image_path}'.")
        return

    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for red color in HSV
    lower_bound1 = np.array([0, 50, 50])
    upper_bound1 = np.array([10, 255, 255])
    lower_bound2 = np.array([160, 50, 50])
    upper_bound2 = np.array([180, 255, 255])

    # Create masks for red color
    mask1 = cv2.inRange(hsv_image, lower_bound1, upper_bound1)
    mask2 = cv2.inRange(hsv_image, lower_bound2, upper_bound2)
    mask = mask1 | mask2

    # Apply the mask to the image
    red_segmented = cv2.bitwise_and(image, image, mask=mask)

    # Save the result
    cv2.imwrite(output_path, red_segmented)
    print(f"Red color segmented image saved to '{output_path}'.")

    # Optional: Display the result
    cv2.imshow('Red Color Segmentation', red_segmented)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Example usage
segment_red_color('images.jpeg', 'red_segmented_image.jpg')
