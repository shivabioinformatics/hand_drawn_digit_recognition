# here i load model and predict custom image

# predict.py

import joblib
import numpy as np
import cv2

# Load model once when this module is imported
model, target_names = joblib.load("digit_model.pkl")

def predict_digit_from_array(img_array):
    """
    img_array: numpy array representing the image in RGB or grayscale
    Returns: predicted digit (int)
    """
    # Convert to grayscale if needed
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    # Resize to 8x8 (sklearn digits size)
    small = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)

    # Invert colors and scale to 0-16
    small = 255 - small
    small = (small / 16).clip(0, 16)

    input_data = small.flatten().reshape(1, -1)

    probs = model.predict_proba(input_data)[0]  # Probability array for all classes
    prediction = model.predict(input_data)[0]
    confidence = probs[prediction]

    return prediction, confidence

def predict_digit_from_file(image_path):
    """
    Load image from path and predict digit.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    return predict_digit_from_array(img)
