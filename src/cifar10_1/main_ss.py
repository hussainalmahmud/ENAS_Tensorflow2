from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
# Now you can use relative imports
from src.cifar10_1 import data_utils
from src import utils
from src.cifar10_1.general_child import GeneralChild
from src.cifar10_1.general_controller import GeneralController
from src.cifar10_1.general_child import GeneralChild
from src.cifar10_1.data_utils import read_data



# Define your training and hyperparameters
num_epochs = 100
batch_size = 128
learning_rate = 0.1



import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10, cifar100
tf.config.run_functions_eagerly(True)

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

    # convert to tensors
    # images["train"] = tf.convert_to_tensor(images["train"], dtype=tf.float32)
    # images["valid"] = tf.convert_to_tensor(images["valid"], dtype=tf.float32)
    # images["test"] = tf.convert_to_tensor(images["test"], dtype=tf.float32)
    images = {
        "train": tf.convert_to_tensor(images["train"], dtype=tf.float32),
        "valid": tf.convert_to_tensor(images["valid"], dtype=tf.float32),
        "test": tf.convert_to_tensor(images["test"], dtype=tf.float32)
    }
    labels = {
        "train": tf.convert_to_tensor(labels["train"], dtype=tf.uint8),
        "valid": tf.convert_to_tensor(labels["valid"], dtype=tf.uint8),
        "test": tf.convert_to_tensor(labels["test"], dtype=tf.uint8)
    }

    return images, labels


# Load CIFAR-10 dataset
# images, labels = data_utils.load_cifar10("data/cifar-10-batches-py")
images, labels = read_data("cifar10")

# Split the dataset into training, validation, and test sets
train_images, train_labels = images["train"], labels["train"]
valid_images, valid_labels = images["valid"], labels["valid"]
test_images, test_labels = images["test"], labels["test"]

# Split the dataset into training, validation, and test sets
# train_images, train_labels, valid_images, valid_labels, test_images, test_labels = data_utils.create_splits(images, labels)

# Create TensorFlow dataset and data iterators
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(batch_size).shuffle(buffer_size=len(train_images))
valid_dataset = tf.data.Dataset.from_tensor_slices((valid_images, valid_labels)).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size)

# Create an instance of the Controller class (you may need to adjust the arguments)
# Define your controller configuration parameters
controller_config = {
    "search_for": "macro",
    "search_whole_channels": False,
    "num_layers": 4,
    "num_branches": 6,
    # Add other configuration parameters as needed
}

# Create an instance of GeneralController with the specified configuration
controller = GeneralController(**controller_config)

# Create an instance of the GeneralChild class (you may need to adjust the arguments)
# child_model = GeneralChild(images, labels, input_images, input_labels)
child_model = GeneralChild(images, labels)

# Connect the controller to the child model
child_model.connect_controller(controller)

# Define loss function and optimizer
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)

# Training loop
for epoch in range(num_epochs):
    for batch_images, batch_labels in train_dataset:
        # Forward pass
        with tf.GradientTape() as tape:
            logits = child_model._model(batch_images, is_training=True)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=batch_labels)
            loss = tf.reduce_mean(loss)

        # Backpropagation
        grads = tape.gradient(loss, child_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, child_model.trainable_variables))

    # Validation
    valid_accuracy = utils.evaluate_accuracy(valid_dataset, child_model)
    print("Epoch {}, Validation Accuracy: {:.4f}".format(epoch + 1, valid_accuracy))

# Test the trained model
test_accuracy = utils.evaluate_accuracy(test_dataset, child_model)
print("Test Accuracy: {:.4f}".format(test_accuracy))
