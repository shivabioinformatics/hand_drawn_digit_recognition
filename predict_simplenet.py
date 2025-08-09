# load model and predict digits

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps
import numpy as np

# Same SimpleNet definition from training
class SimpleNet(torch.nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load model
model = SimpleNet()
model.load_state_dict(torch.load("simplenet.pt"))
model.eval()

# Transform — must match training
transform = transforms.Compose([
    transforms.Grayscale(),       # Ensure grayscale
    transforms.Resize((28, 28)),  # Ensure 28x28
    transforms.ToTensor(),        # Scale to [0,1]
])

# Load your drawn image
image = Image.open("digit.png").convert("L")   # Convert to grayscale
# image = ImageOps.invert(image)                 # Invert if background is white
image = transform(image)
image = image.unsqueeze(0)                     # Add batch dimension

# Predict
with torch.no_grad():
    output = model(image)
    probs = F.softmax(output, dim=1)
    pred_class = torch.argmax(probs, dim=1).item()
    confidence = probs[0, pred_class].item()

print(f"Predicted digit: {pred_class}")
print(f"Confidence: {confidence:.4f}")
print(f"All class probabilities: {probs.numpy()}")
