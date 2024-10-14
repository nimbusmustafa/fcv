import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np


def unsharp_mask(image, kernel_size=(11, 11), sigma=10, amount=3.0):
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
    return sharpened


def compute_gradient(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    gradient_image = np.uint8(magnitude)
    return gradient_image


def apply_box_filter(image, ksize=5):
    return cv2.boxFilter(image, -1, (ksize, ksize))


def apply_gaussian_filter(image, ksize=5, sigma=1.0):
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)


def detect_edges(image, low_threshold=50, high_threshold=150):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    return edges


def detect_edges_custom(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Sobel operators for detecting edges
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Compute the gradient magnitude
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    edges_custom = np.uint8(magnitude)
    return edges_custom


def load_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        global image_path, loaded_image
        image_path = file_path
        loaded_image = cv2.imread(image_path)
        display_image(loaded_image)


def display_image(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    image_tk = ImageTk.PhotoImage(image_pil)
    panel_result.config(image=image_tk)
    panel_result.image = image_tk


def apply_operation():
    if 'image_path' not in globals():
        messagebox.showerror("Error", "Please load an image first.")
        return

    operation = operation_var.get()
    image = loaded_image

    if operation == "Unsharp Mask":
        result_image = unsharp_mask(image)
    elif operation == "Compute Gradient":
        result_image = compute_gradient(image)
    elif operation == "Compare Box & Gaussian Filters":
        box_filtered = apply_box_filter(image, ksize=5)
        gaussian_filtered = apply_gaussian_filter(image, ksize=5, sigma=1.0)

        # Convert images to displayable format
        image_rgb_box = cv2.cvtColor(box_filtered, cv2.COLOR_BGR2RGB)
        image_rgb_gaussian = cv2.cvtColor(gaussian_filtered, cv2.COLOR_BGR2RGB)

        image_pil_box = Image.fromarray(image_rgb_box)
        image_pil_gaussian = Image.fromarray(image_rgb_gaussian)

        image_tk_box = ImageTk.PhotoImage(image_pil_box)
        image_tk_gaussian = ImageTk.PhotoImage(image_pil_gaussian)

        # Display the filtered images side by side
        panel_box.config(image=image_tk_box)
        panel_box.image = image_tk_box
        panel_gaussian.config(image=image_tk_gaussian)
        panel_gaussian.image = image_tk_gaussian

        return

    elif operation == "Detect Edges":
        edges_image = detect_edges(image)
        # Convert edges image to 3-channel to display in the same panel
        edges_image_color = cv2.cvtColor(edges_image, cv2.COLOR_GRAY2BGR)
        display_image(edges_image_color)
        return

    elif operation == "Detect Edges Custom":
        edges_custom_image = detect_edges_custom(image)
        # Convert custom edges image to 3-channel to display in the same panel
        edges_custom_image_color = cv2.cvtColor(edges_custom_image, cv2.COLOR_GRAY2BGR)
        display_image(edges_custom_image_color)
        return

    display_image(result_image)


# Create the main window
root = tk.Tk()
root.title("Image Processing GUI")

# Create and place the buttons and dropdown menu
load_button = tk.Button(root, text="Load Image", command=load_image)
load_button.pack()

# Create a dropdown menu for operations
operation_var = tk.StringVar()
operation_var.set("Select Operation")

operation_menu = ttk.Combobox(root, textvariable=operation_var)
operation_menu['values'] = (
"Unsharp Mask", "Compute Gradient", "Compare Box & Gaussian Filters", "Detect Edges", "Detect Edges Custom")
operation_menu.pack()

apply_button = tk.Button(root, text="Apply", command=apply_operation)
apply_button.pack()

# Create panels for displaying images
panel_result = tk.Label(root)
panel_result.pack()

panel_box = tk.Label(root)
panel_box.pack(side=tk.LEFT)

panel_gaussian = tk.Label(root)
panel_gaussian.pack(side=tk.RIGHT)

# Start the Tkinter event loop
root.mainloop()
