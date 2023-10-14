# 📚 **Dataset Requirement: CIFAR-10**

The Efficient Neural Architecture Search (ENAS) model requires the CIFAR-10 dataset for training and evaluation. Due to its size, it is not included in this repository. Follow the instructions below to download and structure the dataset correctly for use with the model.

## 📥 **Downloading CIFAR-10**

1. Navigate to the [CIFAR-10 Dataset Download Page](https://www.cs.toronto.edu/~kriz/cifar.html).
   
2. Download the `CIFAR-10 python version`, which is typically available as a tar.gz file.

3. Extract the contents of the downloaded file. You should find a folder named `cifar-10-batches-py`.

## 📍 **Placing the Dataset**

1. Ensure you have a folder named `data` in the main project directory.
   
2. Within the `data` folder, create a new folder named `cifar10`.

3. Move all contents from the extracted `cifar-10-batches-py` to the newly created `cifar10` folder.

### 📁 **Final Directory Structure**

Ensure your directory structure looks like this to enable smooth operation of the ENAS model scripts:

```plaintext
├── data
│   └── cifar10
│       ├── data_batch_1
│       ├── data_batch_2
│       ├── data_batch_3
│       ├── data_batch_4
│       ├── data_batch_5
│       └── test_batch
```

## 🚀 **Running the Model**

With the dataset in place, refer back to the main README for instructions on executing the model training and evaluation scripts.
