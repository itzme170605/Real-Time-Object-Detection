# Real-Time American Sign Language (ASL) Recognition

This project implements a real-time American Sign Language (ASL) recognition system using deep learning. The system classifies ASL alphabet letters (A-Z) based on input images and provides efficient performance for on-device inference through model quantization.

---

## Features
- **Dataset Preprocessing**: Organizes and preprocesses a dataset of ASL alphabet images.
- **Model Architecture**: Utilizes a modified ResNet-18 model for ASL letter classification.
- **Quantization**: Optimizes the model for deployment on resource-constrained devices using dynamic quantization.
- **Deployment**: Packages the quantized model and metadata for deployment.
- **Inference**: Includes a utility to predict letters from input images and map predictions to corresponding alphabet letters.

---

## Project Structure
- `train_path` and `test_path`: Directories containing preprocessed training and testing datasets.
- `model`: ResNet-18 model with a modified fully connected layer for 26-class ASL classification.
- `quantized_model`: Quantized version of the model for optimized performance.
- `deployment_package`: Directory containing the packaged model and metadata for deployment.

---

## Requirements
Install the following Python libraries:

```bash
pip install torch torchvision matplotlib pillow
```

---

## Dataset
The dataset is expected to be organized in the following structure:
```
preprocessed_train/
    A/
        image1.png
        image2.png
    B/
        image1.png
        image2.png
    ...
preprocessed_test/
    A/
        image1.png
        image2.png
    ...
```
Each subdirectory corresponds to an ASL letter (A-Z), containing images for that class.

---

## How to Run

### 1. Train and Quantize the Model
The script trains a ResNet-18 model and quantizes it for deployment. The quantized model is saved as `quantized_asl_model.pth`.

```python
# Train the model
train_model(model, train_loader, criterion, optimizer, num_epochs=10)

# Quantize the model
quantized_model = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

# Save the quantized model
torch.save(quantized_model, "quantized_asl_model.pth")
```

### 2. Test the Quantized Model
Evaluate the quantized model's accuracy on the test dataset:

```python
# Test the quantized model
test_quantized_model(quantized_model, test_loader)
```

### 3. Map Predictions to Letters
Map the predicted class index to the corresponding ASL letter:

```python
predicted_class = 5
predicted_letter = map_class_to_letter(predicted_class)
print(f"Predicted class {predicted_class} corresponds to letter: {predicted_letter}")
```

### 4. Prepare for Deployment
Package the quantized model and metadata for deployment:

```python
prepare_model_for_deployment("quantized_asl_model.pth")
```

This creates a `deployment_package` directory containing:
- `quantized_asl_model.pth`: The quantized model.
- `model_metadata.json`: Metadata including input size, number of classes, and framework details.

---

## Performance
- **Accuracy**: Evaluated on the test dataset after training and quantization.
- **Inference Speed**: Real-time performance measured using the packaged model.

---

## Challenges and Decisions
### Challenges
1. **Dataset Preprocessing**:
   - Challenge: The original dataset images had inconsistent sizes and significant background noise.
   - Decision: Resized all images to 224x224 and applied normalization to match ResNet-18 input requirements.

2. **Class Imbalance**:
   - Challenge: Potential for class imbalance across the ASL alphabet dataset.
   - Decision: Augmented the data using techniques like flipping and rotations to ensure sufficient representation for all classes.

3. **Model Selection**:
   - Challenge: Balancing model accuracy and inference speed for real-time performance.
   - Decision: Chose ResNet-18 due to its balance of simplicity and effectiveness, with dynamic quantization to reduce latency.

4. **Quantization Issues**:
   - Challenge: Ensuring the quantized model was correctly saved and loaded for deployment.
   - Decision: Saved the full model object instead of just the `state_dict` to preserve architecture and weights.

5. **Mapping Predictions to Letters**:
   - Challenge: Mapping model outputs (class indices) to corresponding ASL letters.
   - Decision: Implemented a simple dictionary-based mapping from indices (0-25) to letters (A-Z).

### Key Decisions
- Used `torchvision.transforms` for consistent preprocessing.
- Packaged the quantized model with metadata for easy deployment and reuse.
- Tested the model's performance with both original and quantized versions to ensure accuracy was retained.

---

## Acknowledgments
- This project uses PyTorch's `torchvision` library for model architecture and data handling.
- ResNet-18 is used as the base model, pre-trained on ImageNet.
- Kaggle datasets for images of letters in ASL
---

## Future Enhancements
- Extend the system to recognize dynamic gestures (e.g., words or phrases), basically making a video based model.
- Deploy on embedded devices like Raspberry Pi or Jetson Nano.
- Implement additional optimizations for latency and accuracy.

---

