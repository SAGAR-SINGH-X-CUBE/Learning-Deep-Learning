# Neural Network Training Notes

## Neural Network Training (Supervised Learning)

In supervised learning, a neural network learns from **labeled data**. The goal is to find the optimal weights that minimize prediction error.

### Steps for Training a Neural Network

1. **Initialize the Neural Network**
   - The network is initialized with **random weights and biases**.

2. **Feed Forward (Forward Propagation)**
   - Input training samples are passed through the network.
   - The network generates predictions.

3. **Calculate Prediction Error**
   - The difference between the **predicted output** and the **actual output** is calculated.

4. **Calculate Total Error (Loss Function)**
   - A loss function measures the overall error.
   - A common loss function is **Mean Squared Error (MSE)**.

   MSE formula:

   \[
   MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y_i})^2
   \]

   Where:
   - \(y_i\) = actual value  
   - \(\hat{y_i}\) = predicted value  
   - \(n\) = number of samples

5. **Backpropagation**
   - The error is propagated backward through the network.
   - Gradients are calculated for each weight.

6. **Weight Update using Gradient Descent**
   - Weights are updated using gradient descent to reduce the loss.

7. **Repeat for Multiple Epochs**
   - The entire dataset is passed through the network multiple times.
   - One complete pass of the dataset is called **one epoch**.

   Example:
   - If the dataset has **1000 samples**, feeding all samples once = **1 epoch**.

8. **Evaluate the Model**
   - After training, the model is evaluated using **test or validation data**.

---

# Optimization in Neural Networks

Finding the correct weights is **computationally expensive** because neural networks may contain **millions of parameters**.

Therefore, **optimization algorithms** are used to reduce the number of trials needed to reach optimal weights.

---

# Gradient Descent

Gradient Descent is an **optimization technique** used in neural networks and statistical learning to find the optimal weight values that **minimize the prediction error (loss function).**

The goal is to reach the **minimum point of the loss function**.

- Best possible point = **Global Minimum**

### Learning Rate

The **learning rate** is a hyperparameter that controls how much weights change during each update.

- If learning rate is **too high** → training becomes unstable
- If learning rate is **too low** → training becomes very slow

---

# Types of Gradient Descent

## 1. Batch Gradient Descent

Batch Gradient Descent updates weights by computing the gradient using the **entire dataset**.

**Steps:**
1. Use the full dataset
2. Compute total loss
3. Update weights once

**Characteristics**

- Data used per update: Entire dataset
- Convergence speed: Slow
- Computation cost: High
- Memory usage: High
- Stability: Very stable
- Noise: Low

**Example Use Case**

- Small datasets
- Offline machine learning training

---

## 2. Mini-Batch Gradient Descent

Mini-batch gradient descent divides the dataset into **small batches** and updates weights after each batch.

Example:
- Dataset = 10,000 samples
- Batch size = 100
- 100 samples used per update

**Characteristics**

- Data used per update: Small subset (batch)
- Convergence speed: Faster than batch GD
- Computation cost: Moderate
- Memory usage: Moderate
- Stability: Balanced
- Noise: Medium

**Example Use Case**

- Deep learning
- Real-world ML problems
- Training with GPUs

This is the **most commonly used method** in deep learning.

---

## 3. Stochastic Gradient Descent (SGD)

Stochastic Gradient Descent updates weights **after every single training sample**.

**Steps**
1. Take one sample
2. Compute error
3. Update weights
4. Repeat

**Characteristics**

- Data used per update: Single sample
- Convergence speed: Very fast updates
- Computation cost: Low per update
- Memory usage: Low
- Stability: Less stable
- Noise: High (loss fluctuates)

**Example Use Case**

- Very large datasets
- Online learning systems

---

# Comparison of Gradient Descent Methods

| Feature | Batch GD | Mini-Batch GD | Stochastic GD |
|------|------|------|------|
| Data used per update | Entire dataset | Small subset | Single sample |
| Convergence speed | Slow | Fast | Fast but unstable |
| Computation cost | High | Moderate | Low |
| Memory usage | High | Moderate | Low |
| Convergence stability | Very stable | Stable | Noisy |
| Noise in updates | Low | Medium | High |
| Suitable for large datasets | No | Yes | Yes |

---

# Summary

- Neural networks learn by **adjusting weights to minimize prediction error**.
- **Backpropagation + Gradient Descent** are the core mechanisms used during training.
- Different gradient descent methods affect **speed, memory usage, and stability**.
- **Mini-batch gradient descent** is the most commonly used method in modern deep learning systems.
