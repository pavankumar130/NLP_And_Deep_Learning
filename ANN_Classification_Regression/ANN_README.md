# Artificial Neural Networks (ANNs)

This README.md provides an in-depth explanation of **Artificial Neural Networks (ANNs)**, including their structure, mathematical foundations, training algorithms, activation functions, optimizers, regularization, and a sample Python implementation using both TensorFlow/Keras and PyTorch.

---

## 📚 Table of Contents

1. Introduction
2. Biological Inspiration
3. ANN Architecture
   - Neuron structure
   - Layers: Input, Hidden, Output
   - Weights and Biases
4. Mathematical Foundation
   - Forward propagation
   - Activation functions
   - Loss functions
   - Backpropagation
   - Gradient Descent
5. Activation Functions (with pros and cons)
6. Loss Functions
7. Optimizers
8. Regularization Techniques
9. Training and Evaluation
10. Sample Python Code
    - Keras Implementation
    - PyTorch Implementation
11. Common Applications
12. Further Reading & References

---

## 1. Introduction

**Artificial Neural Networks (ANNs)** are computational models inspired by the human brain’s structure. They consist of layers of neurons that transform input data into meaningful output through a series of weighted connections and nonlinear activation functions.

ANNs are the foundation of deep learning and are used for tasks like classification, regression, time-series prediction, and natural language processing.

---

## 2. Biological Inspiration

- The human brain contains billions of neurons connected through synapses.
- Each neuron receives inputs, processes them, and sends output signals to other neurons.
- ANNs simulate this behavior through **artificial neurons** that perform mathematical operations to mimic biological computation.

---

## 3. ANN Architecture

### 🔹 Neuron Structure
Each artificial neuron takes multiple inputs, applies weights, adds a bias, and passes the result through an activation function:

\[ z = \sum_i w_i x_i + b \]
\[ a = f(z) \]

Where:
- \( x_i \): input features
- \( w_i \): weights
- \( b \): bias term
- \( f \): activation function

### 🔹 Layers
1. **Input Layer:** Accepts input features.
2. **Hidden Layers:** Learn feature representations through weighted transformations and nonlinear activations.
3. **Output Layer:** Produces the final prediction (e.g., classification probabilities or regression values).

---

## 4. Mathematical Foundation

### 🔹 Forward Propagation
Each layer computes:
\[ a^{(l)} = f(W^{(l)} a^{(l-1)} + b^{(l)}) \]
Where:
- \( a^{(l-1)} \): activations from the previous layer
- \( W^{(l)} \): weight matrix
- \( b^{(l)} \): bias vector
- \( f \): activation function

### 🔹 Loss Function
Measures the difference between predicted output and true labels.
Examples:
- Mean Squared Error (MSE) for regression
- Cross-Entropy for classification

### 🔹 Backpropagation
- Computes gradients of the loss with respect to weights using the **chain rule**.
- Updates parameters to minimize loss:
  \[ W := W - \eta \frac{\partial L}{\partial W} \]
  Where \( \eta \) is the learning rate.

### 🔹 Gradient Descent
Optimization algorithm that updates weights in the direction of steepest descent of the loss function.

---

## 5. Activation Functions

| Activation | Formula | Advantages | Disadvantages |
|-------------|----------|-------------|----------------|
| **Sigmoid** | \( f(x) = \frac{1}{1 + e^{-x}} \) | Smooth output, interpretable as probability | Vanishing gradients, slow convergence |
| **Tanh** | \( f(x) = \tanh(x) \) | Zero-centered, stronger gradients | Still suffers vanishing gradient |
| **ReLU** | \( f(x) = \max(0, x) \) | Fast, sparse activation | Dying ReLU problem |
| **Leaky ReLU** | \( f(x) = x \) if \(x>0\), else \(0.01x\) | Fixes dying ReLU issue | Slightly more computation |
| **Softmax** | \( f(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}} \) | Multi-class output probabilities | Used only in output layer |

---

## 6. Loss Functions

| Task | Common Loss Function |
|------|-----------------------|
| Classification | Categorical Cross-Entropy, Binary Cross-Entropy |
| Regression | Mean Squared Error (MSE), Mean Absolute Error (MAE) |

Example:
\[ L = -\sum y_i \log(\hat{y_i}) \]

---

## 7. Optimizers

| Optimizer | Description | Pros | Cons |
|------------|-------------|------|------|
| **SGD** | Basic gradient descent | Simple, effective | Sensitive to learning rate |
| **SGD + Momentum** | Adds momentum term to accelerate convergence | Faster training | Requires tuning momentum |
| **Adam** | Adaptive learning rate using moment estimates | Fast, widely used | May overfit or generalize poorly |
| **RMSprop** | Adaptive learning rate per parameter | Stable for RNNs | Slightly slower convergence |

---

## 8. Regularization Techniques

- **L1/L2 Regularization**: Adds penalty to large weights.
- **Dropout**: Randomly drops neurons during training to prevent overfitting.
- **Early Stopping**: Stop training when validation loss stops improving.
- **Batch Normalization**: Stabilizes and speeds up training by normalizing activations.

---

## 9. Training and Evaluation

1. Initialize weights (random or Xavier/He initialization).
2. Forward propagate inputs.
3. Compute loss.
4. Backpropagate errors.
5. Update weights.
6. Repeat for multiple epochs.

Metrics:
- **Accuracy** for classification
- **MSE/MAE** for regression

---

## 10. Sample Python Code

### 🔹 TensorFlow / Keras Example (Classification)

```python
import tensorflow as tf
from tensorflow.keras import layers, models, utils, datasets

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Flatten images
x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)

# One-hot encode labels
y_train = utils.to_categorical(y_train, 10)
y_test = utils.to_categorical(y_test, 10)

# Build ANN model
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.1)

loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy: {acc:.4f}")
```

### 🔹 PyTorch Example

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# Model
class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = ANN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training Loop
for epoch in range(5):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Evaluation
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")
```

---

## 11. Common Applications

- Image classification (when used with CNNs)
- Text classification / sentiment analysis
- Fraud detection
- Regression tasks (e.g., price prediction)
- Time-series forecasting (with RNN variants)

---

## 12. Further Reading

- Goodfellow, Bengio, Courville – *Deep Learning* (MIT Press)
- Andrew Ng’s Deep Learning Specialization (Coursera)
- PyTorch and TensorFlow official documentation

---
