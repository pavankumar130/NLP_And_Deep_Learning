# Deep Learning Concepts and Techniques 🤖

This repository provides a detailed overview of fundamental concepts and techniques in deep learning. Each topic includes in-depth explanations, mathematical expressions, and discussions of key methods to help you build a solid theoretical foundation.

---

## 📑 Index

1. [Introduction to Deep Learning](#1-introduction-to-deep-learning)  
2. [Activation Functions](#2-activation-functions)  
3. [Loss Functions](#3-loss-functions)  
4. [Optimizers](#4-optimizers)  
5. [Weight Initialization Techniques](#5-weight-initialization-techniques)  

---

## 1. Introduction to Deep Learning

**Deep Learning** is a subfield of machine learning that uses multi-layered neural networks to learn from vast amounts of data. The "deep" in deep learning refers to the use of multiple **hidden layers** between the input and output layers. These networks can automatically discover intricate features and representations from raw data, which is a major advantage over traditional machine learning methods that require manual feature engineering.

### How it Works
A deep neural network is essentially a stack of layers, where each layer learns to transform its input data into a more abstract representation.  
For example:
- First layer: recognizes **edges**  
- Next layer: combines edges into **shapes**  
- Deeper layers: combine shapes into **complex objects**  

---

## 2. Activation Functions

An **activation function** introduces **non-linearity** into a neural network, allowing it to learn from complex, non-linear relationships in the data. Without activation functions, a neural network would just be a series of linear transformations.

### Key Activation Functions

- **Sigmoid**  
  $$\sigma(x) = \frac{1}{1 + e^{-x}}$$  

- **Tanh**  
  $$\text{tanh}(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$  

- **ReLU (Rectified Linear Unit)**  
  $$\text{ReLU}(x) = \max(0, x)$$  

- **Leaky ReLU**  
  $$\text{Leaky ReLU}(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha x & \text{if } x \le 0 \end{cases}$$  

- **Parametric ReLU (PReLU)**  
  $$\text{PReLU}(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha_i x & \text{if } x \le 0 \end{cases}$$  

- **Exponential Linear Unit (ELU)**  
  $$\text{ELU}(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha(e^x - 1) & \text{if } x \le 0 \end{cases}$$  

- **Softmax**  
  $$\text{Softmax}(x)_i = \frac{e^{x_i}}{\sum_{j=1}^{K} e^{x_j}}$$  

---

## 3. Loss Functions

The **loss function** measures how well a model's predictions align with the actual data. The training objective is to **minimize loss**.  

### Key Loss Functions

- **Mean Squared Error (MSE)**  
  $$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$  

- **Mean Absolute Error (MAE)**  
  $$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$  

- **Huber Loss**  
  $$L_\delta(y, \hat{y}) = \begin{cases} \frac{1}{2}(y-\hat{y})^2 & \text{if } |y-\hat{y}| \le \delta \\ \delta(|y-\hat{y}| - \frac{1}{2}\delta) & \text{otherwise} \end{cases}$$  

- **Root Mean Squared Error (RMSE)**  
  $$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$  

- **Binary Cross-Entropy**  
  $$\text{BCE} = -[y \log(\hat{y}) + (1-y) \log(1-\hat{y})]$$  

- **Categorical Cross-Entropy**  
  $$\text{CCE} = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)$$  

- **Sparse Categorical Cross-Entropy** → works with integer labels instead of one-hot vectors.  

---

## 4. Optimizers

An **optimizer** updates weights and biases to reduce the loss.  

### Key Optimizers

- **Gradient Descent**  
  $$w_{t+1} = w_t - \eta \cdot \nabla J(w_t)$$  

- **Stochastic Gradient Descent (SGD)** → updates weights per sample.  

- **Mini-Batch SGD** → updates weights per batch.  

- **SGD with Momentum** → adds momentum to speed up convergence.  

- **Adagrad (Adaptive Gradient)** → adaptive learning rates.  

- **RMSprop** → exponential decay of squared gradients.  

- **Adam (Adaptive Moment Estimation)** → combines RMSprop + momentum.  

---

## 5. Weight Initialization Techniques

Proper initialization is crucial to avoid **exploding/vanishing gradients**.  

### Key Techniques

- **Uniform Distribution** → simple random initialization.  

- **Xavier/Glorot Initialization** (for Sigmoid/Tanh):  
  $$\text{Var}(W) = \frac{2}{n_{in} + n_{out}}$$  

- **He Initialization** (for ReLU):  
  $$\text{Var}(W) = \frac{2}{n_{in}}$$  

---
