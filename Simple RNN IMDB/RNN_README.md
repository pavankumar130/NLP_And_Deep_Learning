# Recurrent Neural Networks (RNNs)

This README provides a complete explanation of **Simple Recurrent Neural Networks (RNNs)** — their theory, architecture, mathematical foundation, training process, and sample Python code (Keras & PyTorch). It also includes a detailed comparison with **LSTM** and **GRU** networks.

---

## 📑 Table of Contents

1. [Introduction](#1-introduction)  
2. [RNN Architecture](#2-rnn-architecture)  
3. [Mathematical Formulation](#3-mathematical-formulation)  
4. [Training RNN: Backpropagation Through Time (BPTT)](#4-training-rnn-backpropagation-through-time-bptt)  
5. [Activation Functions](#5-activation-functions)  
6. [Loss Functions](#6-loss-functions)  
7. [Optimizers](#7-optimizers)  
8. [Advantages and Disadvantages](#8-advantages-and-disadvantages)  
9. [Regularization in RNNs](#9-regularization-in-rnns)  
10. [Sample Python Implementations](#10-sample-python-implementations)  
11. [Comparison: Simple RNN vs LSTM vs GRU](#11-comparison-simple-rnn-vs-lstm-vs-gru)  
12. [Applications](#12-applications)  
13. [Further Reading](#13-further-reading)

---

## 1. Introduction

A **Recurrent Neural Network (RNN)** is a type of neural network designed to handle **sequential or time-series data**. Unlike feedforward networks (such as ANN or CNN), RNNs have **feedback connections** allowing them to retain information about previous inputs — forming a kind of **short-term memory**.

RNNs are useful for tasks like:
- Natural Language Processing (NLP)
- Speech Recognition
- Stock Price Prediction
- Time Series Forecasting

---

## 2. RNN Architecture

Each RNN cell takes an input vector at time *t* (\(x_t\)) and the previous hidden state (\(h_{t-1}\)) to produce a new hidden state (\(h_t\)).

### Basic RNN Unit:

\[ h_t = f(W_{xh}x_t + W_{hh}h_{t-1} + b_h) \]
\[ y_t = W_{hy}h_t + b_y \]

Where:
- \( x_t \): Input vector at time *t*
- \( h_t \): Hidden state at time *t*
- \( y_t \): Output at time *t*
- \( W_{xh}, W_{hh}, W_{hy} \): Weight matrices
- \( b_h, b_y \): Bias vectors
- \( f \): Activation function (usually tanh or ReLU)

---

## 3. Mathematical Formulation

Given a sequence of inputs \( x_1, x_2, ..., x_T \):

1. **Hidden State Update:**
   \[ h_t = \tanh(W_{xh}x_t + W_{hh}h_{t-1} + b_h) \]

2. **Output:**
   \[ y_t = \text{softmax}(W_{hy}h_t + b_y) \]

### Intuition:
Each hidden state \( h_t \) depends on both the **current input** and the **previous hidden state**, allowing the network to capture temporal dependencies.

---

## 4. Training RNN: Backpropagation Through Time (BPTT)

Since RNNs have recurrent connections, gradients are computed across **multiple time steps** — this process is called **Backpropagation Through Time (BPTT)**.

During BPTT:
- Errors are propagated backward through time steps.
- Long sequences can cause **vanishing or exploding gradients**.

### Vanishing Gradient:
\( |\frac{\partial h_t}{\partial h_{t-1}}| < 1 \) leads to exponentially small gradients.

### Exploding Gradient:
\( |\frac{\partial h_t}{\partial h_{t-1}}| > 1 \) leads to extremely large gradients.

---

## 5. Activation Functions

| Activation | Formula | Advantages | Disadvantages |
|-------------|----------|-------------|----------------|
| **tanh** | \( \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} \) | Smooth, zero-centered | Vanishing gradient for long sequences |
| **ReLU** | \( f(x) = \max(0, x) \) | Prevents vanishing gradient | Can cause dead neurons |
| **Sigmoid** | \( f(x) = \frac{1}{1 + e^{-x}} \) | Probabilistic interpretation | Saturates easily |

---

## 6. Loss Functions

| Task | Common Loss Function |
|------|-----------------------|
| Classification | Cross-Entropy Loss |
| Regression | Mean Squared Error (MSE) |

Example (Cross-Entropy):
\[ L = -\sum y_t \log(\hat{y_t}) \]

---

## 7. Optimizers

| Optimizer | Description | Pros | Cons |
|------------|-------------|------|------|
| **SGD** | Basic gradient descent | Simple, efficient | May converge slowly |
| **Adam** | Adaptive learning rate | Fast convergence | Slight overfitting risk |
| **RMSProp** | Adapts learning rate per weight | Great for RNNs | Complex tuning |

---

## 8. Advantages and Disadvantages

### ✅ Advantages:
- Handles sequential/time-series data effectively
- Captures temporal dependencies
- Works for variable-length input sequences

### ⚠️ Disadvantages:
- Suffers from vanishing/exploding gradients
- Limited memory for long sequences
- Training is computationally expensive

---

## 9. Regularization in RNNs

- **Dropout:** Randomly drops neurons (including recurrent connections) to avoid overfitting.
- **Gradient Clipping:** Limits gradient values to prevent explosion.
- **Early Stopping:** Stops training when validation loss increases.

---

## 10. Sample Python Implementations

### 🔹 TensorFlow / Keras Example

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

# Load IMDB dataset
max_features = 5000
max_len = 200
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)

# Build RNN model
model = Sequential([
    SimpleRNN(64, input_shape=(max_len,)),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train, epochs=3, batch_size=64, validation_split=0.2)
```

### 🔹 PyTorch Example

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Generate dummy sequential data
x = torch.randn(1000, 10, 8)  # (samples, timesteps, features)
y = torch.randint(0, 2, (1000, 1)).float()

dataset = TensorDataset(x, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # Use last timestep
        return torch.sigmoid(out)

model = SimpleRNN(8, 32, 1)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    for inputs, labels in loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
```

---

## 11. Comparison: Simple RNN vs LSTM vs GRU

| Feature | Simple RNN | LSTM | GRU |
|----------|-------------|------|-----|
| **Memory Capability** | Short-term | Long-term (cell state) | Long-term (simplified gates) |
| **Vanishing Gradient** | Common | Rare | Less common |
| **Training Speed** | Fastest | Slowest | Faster than LSTM |
| **Complexity** | Low | High | Moderate |
| **Number of Gates** | None | Input, Forget, Output | Update, Reset |
| **Best Use Case** | Short sequences | Long sequences (NLP, speech) | Medium-length sequences |

---

## 12. Applications

- Text Generation  
- Sentiment Analysis  
- Time Series Forecasting  
- Speech Recognition  
- Handwriting Recognition

---

## 13. Further Reading

- Goodfellow, Bengio, Courville – *Deep Learning* (MIT Press)  
- Andrej Karpathy – *The Unreasonable Effectiveness of RNNs*  
- TensorFlow & PyTorch Documentation  

---