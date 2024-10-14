import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np

# Fixed 15x15 matrix
fixed_matrix = np.array([
    [  0,  10,  20,  30,  40,  50,  60,  70,  80,  90, 100, 110, 120, 130, 140],
    [ 10,  20,  30,  40,  50,  60,  70,  80,  90, 100, 110, 120, 130, 140, 150],
    [ 20,  30,  40,  50,  60,  70,  80,  90, 100, 110, 120, 130, 140, 150, 160],
    [ 30,  40,  50,  60,  70,  80,  90, 100, 110, 120, 130, 140, 150, 160, 170],
    [ 40,  50,  60,  70,  80,  90, 100, 110, 120, 130, 140, 150, 160, 170, 180],
    [ 50,  60,  70,  80,  90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190],
    [ 60,  70,  80,  90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200],
    [ 70,  80,  90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210],
    [ 80,  90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220],
    [ 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230],
    [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240],
    [110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250],
    [120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250,   0],
    [130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250,   0,  10],
    [140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250,   0,  10,  20]
], dtype=np.uint8)

def apply_box_filter(matrix, ksize=3):
    """Apply a 3x3 box filter to the matrix."""
    return cv2.boxFilter(matrix, -1, (ksize, ksize))

def apply_gaussian_filter(matrix, ksize=3, sigma=1.0):
    """Apply a 3x3 Gaussian filter to the matrix."""
    return cv2.GaussianBlur(matrix, (ksize, ksize), sigma)

def display_matrix(matrix, panel):
    """Display a matrix as an image in the Tkinter panel."""
    matrix_rgb = cv2.cvtColor(matrix, cv2.COLOR_GRAY2RGB)
    image_pil = Image.fromarray(matrix_rgb)
    image_tk = ImageTk.PhotoImage(image_pil)
    panel.config(image=image_tk)
    panel.image = image_tk

def print_matrix():
    """Print the matrix to the console."""
    print("Fixed 15x15 Matrix:")
    print(fixed_matrix)

def apply_filter():
    """Apply the selected filter to the matrix."""
    operation = operation_var.get()
    if operation not in ["Box Filter", "Gaussian Filter"]:
        messagebox.showerror("Error", "Please select a valid filter operation.")
        return

    if not fixed_matrix.size:
        messagebox.showerror("Error", "No matrix to apply filters.")
        return

    if operation == "Box Filter":
        result_matrix = apply_box_filter(fixed_matrix, ksize=3)
        display_matrix(result_matrix, panel_result)
    elif operation == "Gaussian Filter":
        result_matrix = apply_gaussian_filter(fixed_matrix, ksize=3, sigma=1.0)
        display_matrix(result_matrix, panel_result)

# Create the main window
root = tk.Tk()
root.title("Matrix Filter GUI")

# Create and place the buttons and dropdown menu
generate_button = tk.Button(root, text="Generate Matrix", command=lambda: display_matrix(fixed_matrix, panel_matrix))
generate_button.pack()

print_button = tk.Button(root, text="Print Matrix", command=print_matrix)
print_button.pack()

apply_button = tk.Button(root, text="Apply Filter", command=apply_filter)
apply_button.pack()

# Create a dropdown menu for filter operations
operation_var = tk.StringVar()
operation_var.set("Select Filter")

operation_menu = ttk.Combobox(root, textvariable=operation_var)
operation_menu['values'] = ("Box Filter", "Gaussian Filter")
operation_menu.pack()

# Create panels for displaying matrices and results
panel_matrix = tk.Label(root)
panel_matrix.pack(side=tk.LEFT, padx=10)

panel_result = tk.Label(root)
panel_result.pack(side=tk.RIGHT, padx=10)

# Display the initial matrix
display_matrix(fixed_matrix, panel_matrix)

# Start the Tkinter event loop
root.mainloop()
