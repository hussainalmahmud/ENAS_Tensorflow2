import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10, cifar100

def read_data(dataset="cifar10", num_valids=5000):
    """Returns separate NumPy arrays for images and labels with train, valid, and test splits."""
    images = {
        "train": None,
        "valid": None,
        "test": None
    }

    labels = {
        "train": None,
        "valid": None,
        "test": None
    }

    if dataset == "cifar10":
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    elif dataset == "cifar100":
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    else:
        raise ValueError("dataset must be either 'cifar10' or 'cifar100'")

    # Normalize pixel values to the range [0,1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    
    # Compute mean and standard deviation
    mean = np.mean(x_train, axis=(0, 1, 2), keepdims=True)
    std = np.std(x_train, axis=(0, 1, 2), keepdims=True)
    
    # Apply normalization
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    # Splitting out validation data from training data
    validation_length = int(num_valids)
    x_val, y_val = x_train[:validation_length], y_train[:validation_length]
    x_train, y_train = x_train[validation_length:], y_train[validation_length:]

    # Apply the same normalization to validation data
    x_val = (x_val - mean) / std

    # Fill in the dictionaries
    images["train"] = x_train
    images["valid"] = x_val
    images["test"] = x_test
    labels["train"] = y_train
    labels["valid"] = y_val
    labels["test"] = y_test

    return images, labels

# # Call the function to load the CIFAR dataset
# dataset = read_data(dataset="cifar10", num_valids=5000)

# # Access the images and labels dictionaries
# images = dataset[0]
# labels = dataset[1]

# # Print some information to verify that it works
# print("Images - Train Shape:", images["train"].shape)
# print("Labels - Train Shape:", labels["train"].shape)
# print("Images - Validation Shape:", images["valid"].shape)
# print("Labels - Validation Shape:", labels["valid"].shape)
# print("Images - Test Shape:", images["test"].shape)
# print("Labels - Test Shape:", labels["test"].shape)
