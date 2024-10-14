import tkinter as tk
from tkinter import filedialog, Button, Label, Frame, Tk, StringVar
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import os

def apply_smoothen(image):
    return cv2.GaussianBlur(image, (0, 0), 10.0)

def apply_unsharp_masking(image):
    blurred = cv2.GaussianBlur(image, (0, 0), 4.0)
    mask = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
    return mask

def apply_box_filter(image):
    return cv2.boxFilter(image, -1, (5, 5))

def apply_gaussian_filter(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

def apply_canny_edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

def apply_laplacian_edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return cv2.convertScaleAbs(laplacian)

def apply_sobel_edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobelx ** 2 + sobely ** 2)
    return cv2.convertScaleAbs(sobel)

def apply_gradient(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(grad_x ** 2 + grad_y ** 2)
    return cv2.convertScaleAbs(grad)

def apply_median_filter(image):
    return cv2.medianBlur(image, 5)  # Kernel size of 5x5 for median filter

def apply_max_filter(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(image, kernel)  # Max filter using dilation

def apply_min_filter(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(image, kernel)  # Min filter using erosion

def open_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        global original_image
        original_image = cv2.imread(file_path)
        original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        show_image(original_image_rgb, "original")

def process_image(func):
    if original_image is not None:
        original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        processed_image = func(original_image_rgb)
        show_image(processed_image, "processed")
        status_var.set("Processed Image")

def show_image(image, image_type):
    image_pil = Image.fromarray(image)
    image_tk = ImageTk.PhotoImage(image_pil)

    if image_type == "original":
        label_original.config(image=image_tk)
        label_original.image = image_tk
    else:
        label_processed.config(image=image_tk)
        label_processed.image = image_tk

# Setup tkinter window
root = Tk()
root.title("Advanced Image Processing GUI")
root.geometry("1400x1200")

# Initialize global variables
original_image = None

# Create Frames
frame_left = Frame(root, bg="#ffffff", padx=10, pady=10, borderwidth=2, relief="sunken")
frame_left.grid(row=0, column=0, sticky="n", padx=10, pady=10)

frame_center = Frame(root, bg="#f5f5f5", padx=10, pady=10)
frame_center.grid(row=0, column=1, sticky="n", padx=10, pady=10)

frame_right = Frame(root, bg="#ffffff", padx=10, pady=10, borderwidth=2, relief="sunken")
frame_right.grid(row=0, column=2, sticky="n", padx=10, pady=10)

frame_bottom = Frame(root, bg="#f0f0f0", pady=5)
frame_bottom.grid(row=1, column=0, columnspan=3, sticky="ew", padx=10)

# Create and place widgets
label_original = Label(frame_left, bg="#ffffff", borderwidth=2, relief="solid")
label_original.pack()

label_processed = Label(frame_right, bg="#ffffff", borderwidth=2, relief="solid")
label_processed.pack()

btn_open = Button(frame_center, text="Open Image", command=open_image, bg="#4CAF50", fg="white", font=("Arial", 12), relief="raised", padx=10)
btn_open.pack(pady=5)

btn_smoothen = Button(frame_center, text="Smoothen", command=lambda: process_image(apply_smoothen), bg="#2196F3", fg="white", font=("Arial", 12), relief="raised", padx=10)
btn_smoothen.pack(pady=5)

btn_unsharp = Button(frame_center, text="Unsharp Masking", command=lambda: process_image(apply_unsharp_masking), bg="#FF5722", fg="white", font=("Arial", 12), relief="raised", padx=10)
btn_unsharp.pack(pady=5)

btn_box_filter = Button(frame_center, text="Box Filter", command=lambda: process_image(apply_box_filter), bg="#FFC107", fg="black", font=("Arial", 12), relief="raised", padx=10)
btn_box_filter.pack(pady=5)

btn_gaussian_filter = Button(frame_center, text="Gaussian Filter", command=lambda: process_image(apply_gaussian_filter), bg="#9C27B0", fg="white", font=("Arial", 12), relief="raised", padx=10)
btn_gaussian_filter.pack(pady=5)

btn_canny = Button(frame_center, text="Canny Edge", command=lambda: process_image(apply_canny_edge_detection), bg="#673AB7", fg="white", font=("Arial", 12), relief="raised", padx=10)
btn_canny.pack(pady=5)

btn_laplacian = Button(frame_center, text="Laplacian Edge", command=lambda: process_image(apply_laplacian_edge_detection), bg="#3F51B5", fg="white", font=("Arial", 12), relief="raised", padx=10)
btn_laplacian.pack(pady=5)

btn_sobel = Button(frame_center, text="Sobel Edge", command=lambda: process_image(apply_sobel_edge_detection), bg="#009688", fg="white", font=("Arial", 12), relief="raised", padx=10)
btn_sobel.pack(pady=5)

btn_gradient = Button(frame_center, text="Gradient", command=lambda: process_image(apply_gradient), bg="#00BCD4", fg="white", font=("Arial", 12), relief="raised", padx=10)
btn_gradient.pack(pady=5)

btn_median_filter = Button(frame_center, text="Median Filter", command=lambda: process_image(apply_median_filter), bg="#009688", fg="white", font=("Arial", 12), relief="raised", padx=10)
btn_median_filter.pack(pady=5)

btn_max_filter = Button(frame_center, text="Max Filter", command=lambda: process_image(apply_max_filter), bg="#4CAF50", fg="white", font=("Arial", 12), relief="raised", padx=10)
btn_max_filter.pack(pady=5)

btn_min_filter = Button(frame_center, text="Min Filter", command=lambda: process_image(apply_min_filter), bg="#FF5722", fg="white", font=("Arial", 12), relief="raised", padx=10)
btn_min_filter.pack(pady=5)

# Status Bar
status_var = StringVar()
status_var.set("Ready")
status_bar = Label(frame_bottom, textvariable=status_var, relief="sunken", anchor="w", bg="#dcdcdc", padx=10)
status_bar.pack(fill="x")

# Add tooltips
def add_tooltip(widget, text):
    tooltip = tk.Toplevel(widget)
    tooltip.wm_overrideredirect(True)
    tooltip.wm_geometry(f"+{widget.winfo_rootx()}+{widget.winfo_rooty() + widget.winfo_height()}")
    label = tk.Label(tooltip, text=text, background="lightyellow", relief="solid", borderwidth=1)
    label.pack()

    def enter(event):
        tooltip.deiconify()

    def leave(event):
        tooltip.withdraw()

    widget.bind("<Enter>", enter)
    widget.bind("<Leave>", leave)

add_tooltip(btn_open, "Open an image file")
add_tooltip(btn_smoothen, "Apply Gaussian smoothing")
add_tooltip(btn_unsharp, "Apply unsharp masking")
add_tooltip(btn_box_filter, "Apply box filter")
add_tooltip(btn_gaussian_filter, "Apply Gaussian filter")
add_tooltip(btn_canny, "Apply Canny edge detection")
add_tooltip(btn_laplacian, "Apply Laplacian edge detection")
add_tooltip(btn_sobel, "Apply Sobel edge detection")
add_tooltip(btn_gradient, "Apply gradient magnitude")
add_tooltip(btn_median_filter, "Apply median filter")
add_tooltip(btn_max_filter, "Apply max filter")
add_tooltip(btn_min_filter, "Apply min filter")

# Start the GUI event loop
root.mainloop()
