# Fashion-MNIST Image Classification with MLP and CNN

This project explores **neural network architectures and hyperparameter choices** for image classification using the **Fashion-MNIST** dataset.

The project was developed as a deep learning exercise to investigate how different model architectures, numbers of neurons, regularization techniques, optimization algorithms, and convolution kernel sizes affect classification performance.

---

## Project Overview

The main objective of this project is to build and evaluate neural networks for classifying grayscale images of clothing items from the Fashion-MNIST dataset.

The project is organized into several experiments:

1. Building a baseline **Multi-Layer Perceptron (MLP)**
2. Investigating the effect of different numbers of neurons
3. Comparing different architectures with two hidden layers
4. Addressing overfitting using **L2 regularization and Dropout**
5. Comparing different optimization algorithms
6. Building a **Convolutional Neural Network (CNN)**
7. Investigating the effect of different convolution kernel sizes

For each experiment, training and validation loss and accuracy are visualized to compare model behavior.

---

## Dataset

The project uses the **Fashion-MNIST** dataset provided through TensorFlow/Keras.

Fashion-MNIST contains **70,000 grayscale images** of clothing and fashion items:

* 60,000 training images
* 10,000 test images
* Image size: **28 × 28 pixels**
* Number of classes: **10**

Each image belongs to one of the following categories:

| Label | Class       |
| ----: | ----------- |
|     0 | T-shirt/top |
|     1 | Trouser     |
|     2 | Pullover    |
|     3 | Dress       |
|     4 | Coat        |
|     5 | Sandal      |
|     6 | Shirt       |
|     7 | Sneaker     |
|     8 | Bag         |
|     9 | Ankle boot  |

The dataset is loaded directly using:

```python
tensorflow.keras.datasets.fashion_mnist
```

---

## Data Preprocessing

The pixel values are originally represented as integers between 0 and 255.

For the MLP experiments, the images are normalized to the range `[0, 1]`:

```python
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
```

The class labels are converted to one-hot encoded vectors using:

```python
to_categorical()
```

For the CNN experiment, the images are additionally reshaped to include a single grayscale channel:

```text
(28, 28) → (28, 28, 1)
```

---

# Experiments

## Q1 — Baseline MLP

The first experiment establishes a baseline Multi-Layer Perceptron model.

### Architecture

```text
Input: 28 × 28 image
        ↓
Flatten
        ↓
Dense: 128 neurons, ReLU
        ↓
Dense: 10 neurons, Softmax
```

The model is trained using:

* Optimizer: **Adam**
* Loss function: **Categorical Cross-Entropy**
* Batch size: **32**
* Epochs: **10**

Training and validation loss and accuracy are plotted to observe the model's learning behavior.

---

## Q2 — Effect of the Number of Neurons

This experiment investigates how changing the number of neurons in the hidden layer affects MLP performance.

### Q2 — Part A

Four different hidden-layer sizes are evaluated:

```text
100
200
300
400
```

All models use the same general architecture:

```text
Flatten
   ↓
Dense: N neurons, ReLU
   ↓
Dense: 10 neurons, Softmax
```

Validation loss and validation accuracy are compared across the different configurations.

### Q2 — Part B

The experiment is extended to an MLP with **two hidden layers**.

The following combinations of neurons are evaluated:

```text
(256, 128)
(128, 256)
(256, 256)
(128, 128)
(300, 300)
(400, 300)
(400, 400)
```

This experiment investigates how both the number of neurons and their distribution across two hidden layers affect model performance.

---

## Q3 — Dealing with Overfitting

The third experiment investigates techniques for reducing overfitting in deeper MLP architectures.

Two regularization techniques are applied:

* **L2 regularization**
* **Dropout**

The models use:

```text
L2 regularization = 0.001
Dropout rate = 0.2
```

Two architectures are compared:

```text
(256, 256)
(400, 400)
```

The resulting validation loss and accuracy are visualized to examine the effect of regularization on model behavior.

---

## Q4 — Comparing Optimization Algorithms

The fourth experiment compares three commonly used optimization algorithms:

* **SGD**
* **RMSprop**
* **Adam**

All optimizers are evaluated using the same baseline MLP architecture:

```text
Flatten
   ↓
Dense: 128 neurons, ReLU
   ↓
Dense: 10 neurons, Softmax
```

Th
