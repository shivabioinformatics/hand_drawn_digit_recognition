# train a neural network using pytorch 

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define a simple feedforward neural network
import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # Fully connected layer 1
        self.fc2 = nn.Linear(128, 64)       # Fully connected layer 2
        self.fc3 = nn.Linear(64, 10)        # Fully connected layer 3 (output)

    def forward(self, x):
        x = x.view(-1, 28 * 28)             # Flatten image to vector
        x = F.relu(self.fc1(x))             # Apply ReLU to fc1 output - f(x) = max(0, x)
        x = F.relu(self.fc2(x))             # Apply ReLU to fc2 output - f(x) = max(0, x)
        x = self.fc3(x)                     # Final layer (no softmax here)
        return x


# Data loaders
transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_data = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1000)

# Initialize model, loss function, and optimizer
model = SimpleNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(5):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# Evaluate on test set
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        output = model(images)
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")

# Save model
torch.save(model.state_dict(), "mnist_model.pt")
