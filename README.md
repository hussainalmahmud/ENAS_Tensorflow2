# ENAS-TF2: Efficient Neural Architecture Search with TensorFlow 2 🚀

ENAS-TF2 is an efficient implementation of the [ENAS (Efficient Neural Architecture Search)](https://arxiv.org/abs/1802.03268) algorithm (Macro-search) using TensorFlow 2.x. The algorithm facilitates finding optimal neural network architectures in a more computationally efficient manner.

## 🚀 Getting Started
## 1. 🔄 Clone the Repository
```
git clone [REPOSITORY_LINK]
cd [REPOSITORY_NAME]
```
## 2. 🐍 Create a Virtual Environment
```
python3 -m venv env
source env/bin/activate 
```

## 3. 📦 Install Dependencies
Install all necessary packages to run the project using:
```
pip install -r requirements.txt
```

## 4. 📂 Download CIFAR-10 Dataset
Make sure Download the CIFAR-10 dataset from this [link](https://www.cs.toronto.edu/~kriz/cifar.html). Extract and place the data batches in the data/cifar10/ directory as per the project structure.

## 5. 🚄 Run the Scripts
Training
Execute the following script to begin the training:
```
sh scripts/cf10_macro_search.sh
```

## Running the ENAS Script & Understanding Macro Search
When sh scripts/cf10_macro_search.sh is run, the ENAS algorithm proceeds through key stages:

Controller Training: A controller proposes new neural network architectures, which are then trained and validated. The performance of these architectures guides the subsequent proposals from the controller.

Architecture Validation: The controller suggests several architectures that are then validated on a dataset, with their performance metrics reported.

### Example output snippet:
In the ENAS macro search space, each network, comprising 
`N` layers, is described by `N` parts. For instance, in an 8-layer network example from the script output:

```
[4]
[2 1]
[0 1 0]
[5 0 1 0]
[2 1 0 1 0]
[4 1 0 1 0 1]
[0 1 0 1 0 1 0]
[5 0 0 1 0 0 0 0]
val_acc=0.2542
```
The first part `[4]` indicates the operation (e.g., convolution type) at the first layer.
Subsequent parts, such as `[2 1]`, consist of an operation type and a sequence. The sequence, comprised of `0s` and `1s`, indicates whether **a skip connection** is formed from previous layers to the current one. For instance, `[2 1]` implies an operation of type `2` at the second layer and a skip connection from layer `1` to layer `2`.


## To-Do List

- [x] Task 1: Implement Macro-search for `Training` phase
- [ ] Task 2: Implement Macro-search for `Re-training` phase

## 🙏 Acknowledgements

This project is built upon the [official ENAS code](https://github.com/melodyguan/enas) by The TensorFlow Authors, licensed under the [Apache 2.0 License](http://www.apache.org/licenses/LICENSE-2.0), and a TensorFlow 2.x adaptation From [LiuFG](https://github.com/LiuFG/enas-tf2/tree/master). Our implementation, tailored for our research, respects the original licensing and aims to advance the field of neural architecture search.
