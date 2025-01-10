import os
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms, datasets, models
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.quantization import quantize_dynamic


train_path = "preprocessed_train"
test_path = "preprocessed_test"

# Define transformations 
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load datasets
def load_data(data_path):
    dataset = datasets.ImageFolder(data_path, transform=transform)
    return dataset

train_dataset = load_data(train_path)
test_dataset = load_data(test_path)

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the ResNet-18 model
model = models.resnet18(pretrained=True)

# Modify the final layer for ASL classification (26 classes)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 26)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Training function
def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = correct / total

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

# Evaluation function
def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")

# Quantize the model for on-device performance
quantized_model = quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)

# Train the model
num_epochs = 10
train_model(model, train_loader, criterion, optimizer, num_epochs)

# Evaluate the model
evaluate_model(model, test_loader)

# Save the quantized model
torch.save(quantized_model.state_dict(), "quantized_asl_model.pth")
print("Quantized model saved.")
