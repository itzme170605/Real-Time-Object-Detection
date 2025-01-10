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

# Get the class name for index 5
class_name = train_dataset.classes[5]
print(f"Class 5 corresponds to: {class_name}")
