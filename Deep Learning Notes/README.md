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

A deep neural network is essentially a stack of layers, where each layer learns to transform its input data into a more abstract representation. For example, in an image, the first layer might learn to recognize edges, the next layer learns to combine edges into shapes, and subsequent layers learn to combine these shapes into complex objects.

---

## 2. Activation Functions

An **activation function** introduces **non-linearity** into a neural network, allowing it to learn from complex, non-linear relationships in the data. Without activation functions, a neural network would just be a series of linear transformations.

### Key Activation Functions

* **Sigmoid**: Maps any input value to a value between 0 and 1. It was historically popular for its smooth gradient but is now less common due to the **vanishing gradient problem**.
    $$\sigma(x) = \frac{1}{1 + e^{-x}}$$
* **Tanh (Hyperbolic Tangent)**: Similar to Sigmoid but maps values between -1 and 1. It is zero-centered, which can make optimization easier.
    $$\text{tanh}(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$
* **ReLU (Rectified Linear Unit)**: The most widely used activation function. It outputs the input directly if it is positive, otherwise, it outputs zero. It solves the vanishing gradient problem for positive values and is computationally efficient.
    $$\text{ReLU}(x) = \max(0, x)$$
* **Leaky ReLU**: An extension of ReLU that allows a small, non-zero gradient when the input is negative, preventing "dying neurons."
    $$\text{Leaky ReLU}(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha x & \text{if } x \le 0 \end{cases}$$
    where $\alpha$ is a small constant (e.g., 0.01).
* **Parametric ReLU (PReLU)**: An improvement on Leaky ReLU where the slope $\alpha$ for the negative part of the function is learned during training, allowing the model to adapt the slope.
    $$\text{PReLU}(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha_i x & \text{if } x \le 0 \end{cases}$$
* **Exponential Linear Unit (ELU)**: A function that helps push the mean activation towards zero, similar to batch normalization, which can speed up learning. It avoids the dying ReLU problem.
    $$\text{ELU}(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha(e^x - 1) & \text{if } x \le 0 \end{cases}$$
* **Softmax**: Used in the output layer for **multi-class classification**. It converts a vector of numbers into a probability distribution, where the sum of the probabilities equals one.
    $$\text{Softmax}(x)_i = \frac{e^{x_i}}{\sum_{j=1}^{K} e^{x_j}}$$

---

## 3. Loss Functions

The **loss function** measures how well a model's predictions align with the actual data. The primary goal of training a deep learning model is to find the set of weights and biases that **minimizes this loss**. The choice of a loss function depends on the problem type (e.g., regression, classification).

### Key Loss Functions

* **Mean Squared Error (MSE)**: Used for **regression problems**. It calculates the average of the squared differences between the predicted and actual values.
    $$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$
* **Mean Absolute Error (MAE)**: Another **regression** loss function. It measures the average of the absolute differences between predictions and actual values. It is less sensitive to outliers than MSE.
    $$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$
* **Huber Loss**: A loss function that is less sensitive to outliers than MSE and smoother than MAE. It is a quadratic function for small errors and a linear function for large errors.
    $$L_\delta(y, \hat{y}) = \begin{cases} \frac{1}{2}(y-\hat{y})^2 & \text{if } |y-\hat{y}| \le \delta \\ \delta(|y-\hat{y}| - \frac{1}{2}\delta) & \text{otherwise} \end{cases}$$
* **Root Mean Squared Error (RMSE)**: The square root of MSE. It is often preferred over MSE because the resulting error is in the same units as the target variable.
    $$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$
* **Binary Cross-Entropy**: Used for **binary classification problems**. It measures the difference between two probability distributions.
    $$\text{BCE} = -[y \log(\hat{y}) + (1-y) \log(1-\hat{y})]$$
* **Categorical Cross-Entropy**: Used for **multi-class classification** where labels are one-hot encoded.
    $$\text{CCE} = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)$$
* **Sparse Categorical Cross-Entropy**: An alternative to CCE used when labels are integer encoded instead of one-hot encoded. It performs the same computation as CCE but handles the integer format directly.

---

## 4. Optimizers

An **optimizer** is an algorithm that adjusts the neural network's attributes, such as weights and learning rate, to reduce the loss. It guides the training process to find the optimal set of weights. 

### Key Optimizers

* **Gradient Descent**: A fundamental optimization algorithm that updates weights by taking steps proportional to the negative of the gradient of the loss function. It considers the entire dataset for each update, which can be computationally expensive.
    $$w_{t+1} = w_t - \eta \cdot \nabla J(w_t)$$
* **Stochastic Gradient Descent (SGD)**: Updates weights using only one randomly selected training example at a time. This makes it faster than Gradient Descent but can lead to noisy updates.
* **Mini-Batch SGD**: A compromise between Gradient Descent and SGD. It updates weights using a small random subset of the training data (**mini-batch**). This is the most common approach as it balances speed and stability.
* **SGD with Momentum**: Accelerates SGD in the relevant direction and dampens oscillations by adding a fraction of the previous update vector to the current one.
* **Adagrad (Adaptive Gradient)**: An adaptive learning rate optimizer that adjusts the learning rate for each parameter. It scales the learning rate inversely proportional to the sum of the past squared gradients, allowing for larger updates for infrequent parameters.
* **RMSprop (Root Mean Square Propagation)**: An adaptive learning rate optimizer that divides the learning rate by an exponentially decaying average of squared gradients. It is effective in handling non-stationary data.
* **Adam (Adaptive Moment Estimation)**: A highly popular and effective optimizer. It combines the advantages of RMSprop and momentum, using adaptive learning rates for each parameter and leveraging both the first and second moments of the gradients.

---

## 5. Weight Initialization Techniques

**Weight initialization** is the process of setting the initial values of the weights. Proper initialization is crucial as it can significantly affect training speed and prevent problems like **exploding gradients** or **vanishing gradients**.

### Key Techniques

* **Uniform Distribution**: Initializing weights from a uniform distribution with a small range. This is a simple form of random initialization.
* **Xavier/Glorot Initialization**: Designed for layers with **Sigmoid** or **Tanh** activation functions. It initializes weights from a distribution with a specific variance, which helps to keep the variance of the activations consistent across all layers.
    $$\text{Var}(W) = \frac{2}{n_{in} + n_{out}}$$
* **He Initialization**: Designed for layers with **ReLU** activation functions. It sets the variance of the weights based on the number of input neurons, which prevents the dying ReLU problem.
    $$\text{Var}(W) = \frac{2}{n_{in}}$$