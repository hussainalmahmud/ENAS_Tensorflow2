#-*- coding : utf-8 -*-
# coding: utf-8
import os
import sys
# import cPickle as pickle
import pickle as pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import os
import urllib.request
import tarfile

# def download_cifar10_data(data_dir='data'):
#     # Check if the data directory exists, if not, create it
#     if not os.path.exists(data_dir):
#         os.makedirs(data_dir)

#     # Define the URL for CIFAR-10 dataset
#     url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
#     file_name = os.path.join(data_dir, "cifar-10-python.tar.gz")

#     # Check if the CIFAR-10 file has already been downloaded
#     if not os.path.exists(file_name):
#         # If not, download it
#         print("Downloading CIFAR-10 dataset...")
#         urllib.request.urlretrieve(url, file_name)

#     # Extract the contents of the tar.gz file
#     with tarfile.open(file_name, 'r:gz') as tar:
#         tar.extractall(data_dir)

#     print("CIFAR-10 dataset downloaded and extracted to {}.".format(data_dir))

# # Usage:
# download_cifar10_data()


def _read_data(data_path, train_files):
  """Reads CIFAR-10 format data. Always returns NHWC format.

  Returns:
    images: np tensor of size [N, H, W, C]
    labels: np tensor of size [N]
  """
  images, labels = [], []
  for file_name in train_files:
    print (file_name)
    full_name = os.path.join(data_path, file_name)
    with open(full_name,'rb') as finp:
      data = pickle.load(finp, encoding='latin1')
      batch_images = data["data"].astype(np.float32) / 255.0
      batch_labels = np.array(data["labels"], dtype=np.int32)
      images.append(batch_images)
      labels.append(batch_labels)
  images = np.concatenate(images, axis=0)
  labels = np.concatenate(labels, axis=0)
  images = np.reshape(images, [-1, 3, 32, 32])
  images = np.transpose(images, [0, 2, 3, 1])

  return images, labels


def read_data(data_path, num_valids=5000):
  print ("-" * 80)
  print ("Reading data")
  images, labels = {}, {}
  train_files = [
    "data_batch_1",
    "data_batch_2",
    "data_batch_3",
    "data_batch_4",
    "data_batch_5",
  ]
  test_file = [
    "test_batch",
  ]
  images["train"], labels["train"] = _read_data(data_path, train_files)
  # plt.imshow(images['train'][2])
  # plt.show()
  if num_valids:
    images["valid"] = images["train"][-num_valids:]
    labels["valid"] = labels["train"][-num_valids:]
    images["train"] = images["train"][:-num_valids]
    labels["train"] = labels["train"][:-num_valids]
  else:
    images["valid"], labels["valid"] = None, None

  images["test"], labels["test"] = _read_data(data_path, test_file)

  print ("Prepropcess: [subtract mean], [divide std]")
  mean = np.mean(images["train"], axis=(0, 1, 2), keepdims=True)
  std = np.std(images["train"], axis=(0, 1, 2), keepdims=True)

  print ("mean: {}".format(np.reshape(mean * 255.0, [-1])))
  print ("std: {}".format(np.reshape(std * 255.0, [-1])))

  images["train"] = (images["train"] - mean) / std
  if num_valids:
    images["valid"] = (images["valid"] - mean) / std
  images["test"] = (images["test"] - mean) / std

  return images, labels

