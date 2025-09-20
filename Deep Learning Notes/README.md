# Deep Learning Concepts and Techniques 🤖

This repository provides a detailed overview of fundamental concepts and techniques in deep learning. Each topic includes in-depth explanations, mathematical expressions, discussions of key methods, along with their advantages and disadvantages to help you build a solid theoretical foundation.

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

Activation functions introduce **non-linearity**, enabling networks to learn complex mappings.  

### Key Activation Functions

- **Sigmoid**  
  Formula: σ(x) = 1 / (1 + e^(-x))  
  - ✅ Advantage: Outputs between 0 and 1 → good for probabilities.  
  - ❌ Disadvantage: Vanishing gradient problem, not zero-centered.  

- **Tanh**  
  Formula: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))  
  - ✅ Advantage: Zero-centered, stronger gradients than sigmoid.  
  - ❌ Disadvantage: Still suffers from vanishing gradients.  

- **ReLU (Rectified Linear Unit)**  
  Formula: ReLU(x) = max(0, x)  
  - ✅ Advantage: Simple, efficient, reduces vanishing gradient.  
  - ❌ Disadvantage: Dying ReLU problem (neurons stuck at 0).  

- **Leaky ReLU**  
  Formula:  
  ```
  Leaky ReLU(x) = x       if x > 0  
                  αx      if x ≤ 0
  ```  
  - ✅ Advantage: Solves dying ReLU by allowing small negative slope.  
  - ❌ Disadvantage: α is fixed and may not be optimal.  

- **Parametric ReLU (PReLU)**  
  Formula:  
  ```
  PReLU(x) = x       if x > 0  
             αᵢx     if x ≤ 0
  ```  
  - ✅ Advantage: Learns negative slope during training.  
  - ❌ Disadvantage: More parameters → risk of overfitting.  

- **Exponential Linear Unit (ELU)**  
  Formula:  
  ```
  ELU(x) = x                if x > 0  
          α(e^x − 1)        if x ≤ 0
  ```  
  - ✅ Advantage: Mean activations closer to 0 → faster learning.  
  - ❌ Disadvantage: Slightly more computationally expensive.  

- **Softmax**  
  Formula:  
  ```
  Softmax(x)_i = exp(x_i) / Σ exp(x_j),  j = 1...K
  ```  
  - ✅ Advantage: Outputs probabilities for multi-class classification.  
  - ❌ Disadvantage: Computationally expensive for large K.  

---

## 3. Loss Functions

The **loss function** measures how well a model's predictions align with the actual values. Training aims to **minimize loss**.

### Key Loss Functions

- **Mean Squared Error (MSE)**  
  Formula: MSE = (1/n) Σ (yᵢ − ŷᵢ)²  
  - ✅ Advantage: Smooth and differentiable.  
  - ❌ Disadvantage: Sensitive to outliers.  

- **Mean Absolute Error (MAE)**  
  Formula: MAE = (1/n) Σ |yᵢ − ŷᵢ|  
  - ✅ Advantage: Robust to outliers.  
  - ❌ Disadvantage: Not differentiable at 0.  

- **Huber Loss**  
  Formula: Combines MSE (small errors) and MAE (large errors).  
  - ✅ Advantage: Balances robustness and smoothness.  
  - ❌ Disadvantage: Requires tuning of δ parameter.  

- **Root Mean Squared Error (RMSE)**  
  Formula: RMSE = √MSE  
  - ✅ Advantage: Error in same units as target.  
  - ❌ Disadvantage: Still sensitive to outliers.  

- **Binary Cross-Entropy (BCE)**  
  Formula: BCE = −[y log(ŷ) + (1 − y) log(1 − ŷ)]  
  - ✅ Advantage: Well-suited for binary classification.  
  - ❌ Disadvantage: Requires careful handling of numerical stability.  

- **Categorical Cross-Entropy (CCE)**  
  Formula: CCE = −Σ yᵢ log(ŷᵢ)  
  - ✅ Advantage: Standard for multi-class problems.  
  - ❌ Disadvantage: Requires one-hot encoding.  

- **Sparse Categorical Cross-Entropy**  
  - ✅ Advantage: Avoids one-hot encoding (uses integer labels).  
  - ❌ Disadvantage: Limited to categorical targets.  

---

## 4. Optimizers

Optimizers update weights to reduce loss.  

### Key Optimizers

- **Gradient Descent**  
  Formula: w(t+1) = w(t) − η ∇J(w)  
  - ✅ Advantage: Theoretical foundation, guaranteed convergence with proper step size.  
  - ❌ Disadvantage: Very slow for large datasets.  

- **Stochastic Gradient Descent (SGD)**  
  - ✅ Advantage: Faster updates, less memory use.  
  - ❌ Disadvantage: Noisy updates, may oscillate.  

- **Mini-Batch SGD**  
  - ✅ Advantage: Balance of speed and stability.  
  - ❌ Disadvantage: Requires choosing batch size.  

- **SGD with Momentum**  
  - ✅ Advantage: Accelerates convergence, reduces oscillations.  
  - ❌ Disadvantage: Extra hyperparameter (momentum).  

- **Adagrad**  
  - ✅ Advantage: Adapts learning rate for each parameter.  
  - ❌ Disadvantage: Learning rate shrinks too much over time.  

- **RMSprop**  
  - ✅ Advantage: Good for non-stationary problems, stable.  
  - ❌ Disadvantage: Still needs hyperparameter tuning.  

- **Adam**  
  - ✅ Advantage: Combines momentum + adaptive rates → fast and popular.  
  - ❌ Disadvantage: Sometimes overfits, may generalize poorly.  

---

## 5. Weight Initialization Techniques

Proper initialization prevents **vanishing/exploding gradients**.  

### Key Techniques

- **Uniform Distribution**  
  - ✅ Advantage: Simple and fast.  
  - ❌ Disadvantage: May cause unstable gradients.  

- **Xavier/Glorot Initialization** (for Sigmoid/Tanh)  
  Formula: Var(W) = 2 / (n_in + n_out)  
  - ✅ Advantage: Balances gradients across layers.  
  - ❌ Disadvantage: Not optimal for ReLU networks.  

- **He Initialization** (for ReLU)  
  Formula: Var(W) = 2 / n_in  
  - ✅ Advantage: Prevents dying ReLU problem.  
  - ❌ Disadvantage: May still suffer if architecture is very deep.  

---
