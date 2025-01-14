import os
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms, datasets, models
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.quantization import quantize_dynamic

# Define dataset paths (replace with your actual paths)
train_path = "preprocessed_train"
test_path = "preprocessed_test"

# Define transformations (already applied during preprocessing)
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

# Modify the final layer for ASL classification (27 classes)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 27)

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Quantize the model for on-device performance
quantized_model = quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)

# Test Quantized Model
def test_quantized_model(quantized_model, dataloader):
    quantized_model.eval()
    quantized_model.to(device)
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = quantized_model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"Quantized Model Test Accuracy: {accuracy:.4f}")

# Evaluate the quantized model
test_quantized_model(quantized_model, test_loader)

# Save the quantized model
quantized_model_path = "quantized_asl_model.pth"
torch.save(quantized_model.state_dict(), quantized_model_path)
print(f"Quantized model saved to {quantized_model_path}")

# Deployment Preparation
def prepare_model_for_deployment(model_path, output_dir="deployment_package"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the model state dict
    deployed_model_path = os.path.join(output_dir, "quantized_asl_model.pth")
    os.rename(model_path, deployed_model_path)

    # Save metadata
    metadata = {
        "input_size": [1, 3, 224, 224],
        "output_classes": 26,
        "framework": "PyTorch",
        "quantized": True
    }
    with open(os.path.join(output_dir, "model_metadata.json"), "w") as f:
        import json
        json.dump(metadata, f)

    print(f"Model and metadata prepared in {output_dir}")

# Prepare the model for deployment
prepare_model_for_deployment(quantized_model_path)
