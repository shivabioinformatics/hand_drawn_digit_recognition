# here i load model and predict custom image

import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt

def preprocess_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    
    img_resized = cv2.resize(img, (8, 8), interpolation=cv2.INTER_AREA)     # sklearn digits are 8x8
    img_resized = 255 - img_resized      # MNIST uses white digits on black bg
    img_normalized = img_resized / 16.0 # sklearn digits use values 0–16
    img_flattened = img_normalized.flatten()

    return img_flattened

def predict_digit(image_path):
    model, target_names = joblib.load("digit_model.pkl")
    input_data = preprocess_image(image_path)
    prediction = model.predict([input_data])[0]    
    proba = model.predict_proba([input_data])[0]
    confidence = np.max(proba) * 100
    print(f"🔢 Predicted digit: {prediction} ({confidence:.1f}% confidence)")


    # Optional: Show processed image
    plt.imshow(input_data.reshape(8, 8), cmap='gray')
    plt.title(f"Predicted: {prediction}")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    predict_digit("digit.png")
