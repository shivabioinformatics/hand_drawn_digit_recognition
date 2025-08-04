# gui app to draw and classify hand-written digits
# using tkinter for GUI and PIL for drawing


import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import joblib
import os
from model import train_and_save_model

# Load model
if not os.path.exists("digit_model.pkl"):
    print("Model not found — training a new one.")
    train_and_save_model()
model, _ = joblib.load("digit_model.pkl")

# Constants
CANVAS_SIZE = 200
DIGIT_SIZE = 8
BRUSH_SIZE = 12

# Set up window
root = tk.Tk()
root.title("HandDigit Classifier")

# Set up drawing canvas
canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="white")
canvas.pack()

# Internal image used for prediction
image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=255)
draw = ImageDraw.Draw(image)

def paint(event):
    # Draw on visible canvas
    x, y = event.x, event.y
    canvas.create_oval(x - BRUSH_SIZE, y - BRUSH_SIZE, x + BRUSH_SIZE, y + BRUSH_SIZE, fill="black", outline="black")

    # Draw on internal image buffer
    draw.ellipse([x - BRUSH_SIZE, y - BRUSH_SIZE, x + BRUSH_SIZE, y + BRUSH_SIZE], fill=0)

def clear_canvas():
    canvas.delete("all")
    draw.rectangle([0, 0, CANVAS_SIZE, CANVAS_SIZE], fill=255)
    result_label.config(text="Draw a digit")

def predict():
    img_resized = image.resize((DIGIT_SIZE, DIGIT_SIZE), Image.Resampling.LANCZOS)
    img_inverted = ImageOps.invert(img_resized)
    arr = np.array(img_inverted, dtype=np.float32)
    arr = arr / 16.0
    flat = arr.flatten()

    prediction = model.predict([flat])[0]
    result_label.config(text=f"🔢 Predicted: {prediction}")

# Bind mouse events
canvas.bind("<B1-Motion>", paint)

# Buttons
btn_frame = tk.Frame(root)
btn_frame.pack(pady=5)

tk.Button(btn_frame, text="Predict", command=predict).pack(side=tk.LEFT, padx=10)
tk.Button(btn_frame, text="Clear", command=clear_canvas).pack(side=tk.LEFT, padx=10)

# Prediction label
result_label = tk.Label(root, text="Draw a digit", font=("Helvetica", 16))
result_label.pack(pady=10)

root.mainloop()
