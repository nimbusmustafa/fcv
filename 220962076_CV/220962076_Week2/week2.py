import cv2
import numpy as np
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
from PIL import Image, ImageTk

# Function to plot histograms
def plot_histograms(original_image, processed_image):
    original_hist, bins = np.histogram(original_image.flatten(), 256, [0, 256])
    processed_hist, bins = np.histogram(processed_image.flatten(), 256, [0, 256])

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Original Image Histogram')
    plt.plot(original_hist, color='black')
    plt.xlim([0, 256])

    plt.subplot(1, 2, 2)
    plt.title('Processed Image Histogram')
    plt.plot(processed_hist, color='black')
    plt.xlim([0, 256])

    plt.show()

# Function to plot RGB histograms
def plot_rgb_histograms(input_image, reference_image, output_image):
    input_channels = cv2.split(input_image)
    reference_channels = cv2.split(reference_image)
    output_channels = cv2.split(output_image)

    plt.figure(figsize=(18, 12))
    colors = ('Red', 'Green', 'Blue')

    for i, color in enumerate(colors):
        plt.subplot(3, 3, i + 1)
        plt.title(f'{color} Channel - Input Image')
        plt.hist(input_channels[i].ravel(), bins=256, color=color.lower())
        plt.xlim([0, 256])

        plt.subplot(3, 3, i + 4)
        plt.title(f'{color} Channel - Reference Image')
        plt.hist(reference_channels[i].ravel(), bins=256, color=color.lower())
        plt.xlim([0, 256])

        plt.subplot(3, 3, i + 7)
        plt.title(f'{color} Channel - Output Image')
        plt.hist(output_channels[i].ravel(), bins=256, color=color.lower())
        plt.xlim([0, 256])

    plt.tight_layout()
    plt.show()

# Function to find and annotate brightest and darkest spots
def annotate_brightest_darkest(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray_image)

    annotated_image = image.copy()
    cv2.rectangle(annotated_image, (min_loc[0] - 10, min_loc[1] - 10), (min_loc[0] + 10, min_loc[1] + 10), (0, 0, 255), 2)  # Red rectangle for darkest spot
    cv2.rectangle(annotated_image, (max_loc[0] - 10, max_loc[1] - 10), (max_loc[0] + 10, max_loc[1] + 10), (0, 255, 0), 2)  # Green rectangle for brightest spot

    cv2.putText(annotated_image, 'Darkest Spot', min_loc, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(annotated_image, 'Brightest Spot', max_loc, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return annotated_image, min_loc, max_loc, min_val, max_val

# Function for image negative
def image_negative(image):
    return 255 - image

# Function for log transform
def log_transform(image):
    c = 255 / np.log(1 + np.max(image))
    log_transformed = c * np.log(1 + image)
    return np.uint8(log_transformed)

# Function for power-law (gamma) transform
def power_law_transform(image, gamma):
    normalized_image = image / 255.0
    power_law_transformed = (normalized_image ** gamma) * 255
    return np.uint8(power_law_transformed)

# Function for piecewise linear transformation
def piecewise_linear_transform(image):
    breakpoints = [0, 64, 128, 192, 255]
    values = [0, 64, 128, 192, 255]

    lookup_table = np.interp(np.arange(256), breakpoints, values)
    piecewise_transformed = lookup_table[image]
    return np.uint8(piecewise_transformed)

# Function for histogram equalization
def histogram_equalization(image_path, output_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found or unable to load.")

    equalized_image = cv2.equalizeHist(image)
    cv2.imwrite(output_path, equalized_image)

    plot_histograms(image, equalized_image)

# Function for histogram specification
def histogram_specification(input_image_path, reference_image_path, output_path):
    input_image = cv2.imread(input_image_path)
    reference_image = cv2.imread(reference_image_path)

    if input_image is None or reference_image is None:
        raise ValueError("One or both images not found or unable to load.")

    input_image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    reference_image_rgb = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)

    input_channels = cv2.split(input_image_rgb)
    reference_channels = cv2.split(reference_image_rgb)

    output_image_rgb = np.zeros_like(input_image_rgb)

    for i in range(3):
        input_hist, bins = np.histogram(input_channels[i].flatten(), 256, [0, 256])
        reference_hist, bins = np.histogram(reference_channels[i].flatten(), 256, [0, 256])

        input_cdf = input_hist.cumsum()
        reference_cdf = reference_hist.cumsum()

        input_cdf_normalized = input_cdf * 255 / input_cdf[-1]
        reference_cdf_normalized = reference_cdf * 255 / reference_cdf[-1]

        lookup_table = np.interp(input_cdf_normalized, reference_cdf_normalized, np.arange(256))
        output_image_rgb[:, :, i] = lookup_table[input_channels[i].flatten()].reshape(input_channels[i].shape).astype(np.uint8)

    cv2.imwrite(output_path, cv2.cvtColor(output_image_rgb, cv2.COLOR_RGB2BGR))

    plot_rgb_histograms(input_image_rgb, reference_image_rgb, output_image_rgb)

# Function for resizing image
def resize_image(image_path, output_path, scale_percent):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to load.")

    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    cv2.imwrite(output_path, resized_image)

# Function for cropping image
def crop_image(image_path, output_path, x, y, w, h):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to load.")

    cropped_image = image[y:y + h, x:x + w]
    cv2.imwrite(output_path, cropped_image)

# Function to plot a single histogram
def plot_histogram(image):
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])

    plt.figure(figsize=(6, 6))
    plt.title('Image Histogram')
    plt.plot(hist, color='black')
    plt.xlim([0, 256])
    plt.show()

# Tkinter GUI Application
class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing App")
        self.create_widgets()

    def create_widgets(self):
        # Create a stylish font for labels
        font = ('Helvetica', 12)

        # Create top panel for dropdown menu
        self.top_frame = tk.Frame(self.root, bg='#f0f0f0')
        self.top_frame.pack(side=tk.TOP, fill=tk.X, padx=20, pady=10)

        # Create left panel for buttons
        self.left_frame = tk.Frame(self.root, bg='#f0f0f0')
        self.left_frame.pack(side=tk.LEFT, padx=20, pady=20, fill=tk.BOTH, expand=True)

        # Create right panel for displaying images
        self.right_frame = tk.Frame(self.root, bg='#f0f0f0')
        self.right_frame.pack(side=tk.RIGHT, padx=20, pady=20, fill=tk.BOTH, expand=True)

        # Top panel: Dropdown menu
        self.option_label = tk.Label(self.top_frame, text="Choose an Operation", font=font, bg='#f0f0f0')
        self.option_label.pack(pady=10)

        self.option_menu = ttk.Combobox(self.top_frame, values=[
            'Histogram Equalization', 'Histogram Specification', 'Resize Image', 'Crop Image',
            'Image Negative', 'Log Transform', 'Power-Law Transform', 'Piecewise Linear Transform',
            'Find Brightest and Darkest Spot'], font=font)
        self.option_menu.pack(pady=10, fill=tk.X)
        self.option_menu.bind('<<ComboboxSelected>>', self.on_option_select)

        # Left panel: Image input and reference buttons
        self.load_image_button = tk.Button(self.left_frame, text="Load Input Image", command=self.load_image, font=font)
        self.load_image_button.pack(pady=10)

        self.load_reference_button = tk.Button(self.left_frame, text="Load Reference Image", command=self.load_reference_image, font=font)
        self.load_reference_button.pack(pady=10)

        # Left panel: Placeholder for input and reference images
        self.input_image_label = tk.Label(self.left_frame, text="Input Image", bg='#f0f0f0')
        self.input_image_label.pack(pady=10)
        self.input_image_display = tk.Label(self.left_frame, bg='#f0f0f0')
        self.input_image_display.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.reference_image_label = tk.Label(self.left_frame, text="Reference Image", bg='#f0f0f0')
        self.reference_image_label.pack(pady=10)
        self.reference_image_display = tk.Label(self.left_frame, bg='#f0f0f0')
        self.reference_image_display.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Right panel: Display processed images
        self.processed_image_label = tk.Label(self.right_frame, text="Processed Image", bg='#f0f0f0')
        self.processed_image_label.pack(pady=10)
        self.processed_image_display = tk.Label(self.right_frame, bg='#f0f0f0')
        self.processed_image_display.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Initialize image paths
        self.image_path = None
        self.reference_image_path = None

    def load_image(self):
        self.image_path = 'inp.png'
        if self.image_path:
            self.display_image(self.image_path, 'input')

    def load_reference_image(self):
        self.reference_image_path = 'ref.png'
        if self.reference_image_path:
            self.display_image(self.reference_image_path, 'reference')

    def display_image(self, image_path, image_type):
        cv_image = cv2.imread(image_path)
        if cv_image is not None:
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            tk_image = ImageTk.PhotoImage(pil_image)

            if image_type == 'input':
                self.input_image_display.config(image=tk_image)
                self.input_image_display.image = tk_image

            elif image_type == 'reference':
                self.reference_image_display.config(image=tk_image)
                self.reference_image_display.image = tk_image

            elif image_type == 'processed':
                self.processed_image_display.config(image=tk_image)
                self.processed_image_display.image = tk_image

    def on_option_select(self, event):
        selected_option = self.option_menu.get()

        if not self.image_path:
            messagebox.showwarning("Warning", "Please load an input image first.")
            return

        if selected_option == 'Histogram Equalization':
            self.perform_histogram_equalization()

        elif selected_option == 'Histogram Specification':
            if not self.reference_image_path:
                messagebox.showwarning("Warning", "Please load a reference image first.")
                return
            self.perform_histogram_specification()

        elif selected_option == 'Resize Image':
            self.perform_resize()

        elif selected_option == 'Crop Image':
            self.perform_crop()

        elif selected_option == 'Image Negative':
            self.perform_image_negative()

        elif selected_option == 'Log Transform':
            self.perform_log_transform()

        elif selected_option == 'Power-Law Transform':
            self.perform_power_law_transform()

        elif selected_option == 'Piecewise Linear Transform':
            self.perform_piecewise_linear_transform()

        elif selected_option == 'Find Brightest and Darkest Spot':
            self.perform_find_brightest_darkest_spot()

    def perform_histogram_equalization(self):
        output_path = 'equalized_image.jpg'
        histogram_equalization(self.image_path, output_path)
        self.display_image(output_path, 'processed')

    def perform_histogram_specification(self):
        output_path = 'specified_image.jpg'
        histogram_specification(self.image_path, self.reference_image_path, output_path)
        self.display_image(output_path, 'processed')

    def perform_resize(self):
        scale_percent = simpledialog.askinteger("Input", "Enter the scale percent:", minvalue=1, maxvalue=100)
        if scale_percent is None:
            return
        output_path = 'resized_image.jpg'
        resize_image(self.image_path, output_path, scale_percent)
        self.display_image(output_path, 'processed')

    def perform_crop(self):
        x = simpledialog.askinteger("Input", "Enter x-coordinate:", minvalue=0)
        y = simpledialog.askinteger("Input", "Enter y-coordinate:", minvalue=0)
        w = simpledialog.askinteger("Input", "Enter width:", minvalue=1)
        h = simpledialog.askinteger("Input", "Enter height:", minvalue=1)
        if None in (x, y, w, h):
            return
        output_path = 'cropped_image.jpg'
        crop_image(self.image_path, output_path, x, y, w, h)
        self.display_image(output_path, 'processed')

    def perform_image_negative(self):
        image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            messagebox.showerror("Error", "Image not found or unable to load.")
            return
        negative_image = image_negative(image)
        output_path = 'negative_image.jpg'
        cv2.imwrite(output_path, negative_image)
        self.display_image(output_path, 'processed')

    def perform_log_transform(self):
        image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            messagebox.showerror("Error", "Image not found or unable to load.")
            return
        log_transformed_image = log_transform(image)
        output_path = 'log_transformed_image.jpg'
        cv2.imwrite(output_path, log_transformed_image)
        self.display_image(output_path, 'processed')

    def perform_power_law_transform(self):
        gamma = simpledialog.askfloat("Input", "Enter the gamma value:", minvalue=0.1, maxvalue=5.0)
        if gamma is None:
            return
        image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            messagebox.showerror("Error", "Image not found or unable to load.")
            return
        power_law_image = power_law_transform(image, gamma)
        output_path = 'power_law_transformed_image.jpg'
        cv2.imwrite(output_path, power_law_image)
        self.display_image(output_path, 'processed')

    def perform_piecewise_linear_transform(self):
        image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            messagebox.showerror("Error", "Image not found or unable to load.")
            return
        piecewise_image = piecewise_linear_transform(image)
        output_path = 'piecewise_linear_transformed_image.jpg'
        cv2.imwrite(output_path, piecewise_image)
        self.display_image(output_path, 'processed')

    def perform_find_brightest_darkest_spot(self):
        image = cv2.imread(self.image_path)
        if image is None:
            messagebox.showerror("Error", "Image not found or unable to load.")
            return
        annotated_image, min_loc, max_loc, min_val, max_val = annotate_brightest_darkest(image)
        output_path = 'annotated_image.jpg'
        cv2.imwrite(output_path, annotated_image)
        self.display_image(output_path, 'processed')
        messagebox.showinfo("Brightest and Darkest Spot",
                            f"Darkest Spot: {min_loc}, Value: {min_val}\n"
                            f"Brightest Spot: {max_loc}, Value: {max_val}")

# Run the application
root = tk.Tk()
app = ImageProcessingApp(root)
root.mainloop()
