from EDA import *

# Perform EDA
print("Train Class Distribution:", get_class_distribution(preprocessed_train_path))
print("Test Class Distribution:", get_class_distribution(preprocessed_test_path))
visualize_samples(preprocessed_train_path)