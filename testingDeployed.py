import json
import os
import time
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms, datasets, models
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.quantization import quantize_dynamic

#tests the deployed model
# Load model and metadata
deployment_dir = "deployment_package"
model_path = f"{deployment_dir}/quantized_asl_model.pth"
metadata_path = f"{deployment_dir}/model_metadata.json"

# Load metadata
with open(metadata_path, "r") as f:
    metadata = json.load(f)

# Model and input specifications
input_size = metadata["input_size"]
output_classes = metadata["output_classes"]

# Load the quantized model
# model = models.resnet18()
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, output_classes)
# model.load_state_dict(torch.load(model_path))
# model.eval()
# Load the quantized model as a full object
# Move model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(model_path, map_location=device)
model.eval()
model.to(device)



# Preprocess the test image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load and preprocess a sample image
image_path = "preprocessed_test/A/0bccc8ff-d22a-4355-a517-aa552ea756a3.rgb_0000.png"
image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)

# Perform inference and measure latency
start_time = time.time()
with torch.no_grad():
    output = model(input_tensor)
end_time = time.time()

_, predicted_class = torch.max(output, 1)
print(f"Predicted Class: {predicted_class.item()}")
print(f"Inference Time: {end_time - start_time:.4f} seconds")
