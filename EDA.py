import os
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from collections import Counter

# Define dataset paths (replace with your actual paths)
train_path = "Test_Alphabet"
test_path = "Train_Alphabet"

# Function to get class distribution
def get_class_distribution(data_path):
    classes = []
    for class_dir in os.listdir(data_path):
        if os.path.isdir(os.path.join(data_path, class_dir)):
            num_images = len(os.listdir(os.path.join(data_path, class_dir)))
            classes.append((class_dir, num_images))
    return dict(classes)

# Function to visualize random samples from each class
def visualize_samples(data_path, num_samples=3):
    class_dirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    fig, axes = plt.subplots(len(class_dirs), num_samples, figsize=(num_samples * 3, len(class_dirs) * 3))

    for i, class_dir in enumerate(sorted(class_dirs)):
        class_path = os.path.join(data_path, class_dir)
        sample_images = os.listdir(class_path)[:num_samples]

        for j, img_name in enumerate(sample_images):
            img_path = os.path.join(class_path, img_name)
            img = Image.open(img_path)
            axes[i, j].imshow(img)
            axes[i, j].axis('off')
            if j == 0:
                axes[i, j].set_title(class_dir)

    plt.tight_layout()
    plt.show()

# Perform EDA
print("Train Class Distribution:", get_class_distribution(train_path))
print("Test Class Distribution:", get_class_distribution(test_path))
visualize_samples(train_path)

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to preprocess and save images
def preprocess_and_save(data_path, save_path):
    os.makedirs(save_path, exist_ok=True)

    for class_dir in os.listdir(data_path):
        class_path = os.path.join(data_path, class_dir)
        save_class_path = os.path.join(save_path, class_dir)
        os.makedirs(save_class_path, exist_ok=True)

        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                save_img_path = os.path.join(save_class_path, img_name)

                try:
                    img = Image.open(img_path)
                    img_transformed = transform(img)
                    img_transformed = transforms.ToPILImage()(img_transformed)
                    img_transformed.save(save_img_path)
                except Exception as e:
                    print(f"Error processing {img_name}: {e}")

# Preprocess and save datasets
preprocessed_train_path = "preprocessed_train"
preprocessed_test_path = "preprocessed_test"

preprocess_and_save(train_path, preprocessed_train_path)
preprocess_and_save(test_path, preprocessed_test_path)
