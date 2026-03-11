# Deep Learning & PyTorch Notes

# 1. Deep Learning

Deep Learning is a subset of **Machine Learning** that uses **Artificial Neural Networks (ANNs)** with multiple layers to learn patterns from large amounts of data.

Deep learning models automatically extract features from data instead of relying heavily on manual feature engineering.

Example applications:
- Image recognition
- Speech recognition
- Natural language processing
- Self-driving cars
- Generative AI (ChatGPT, image generation)

---

# 2. When to Use Statistical Machine Learning vs Deep Learning

## Use Statistical Machine Learning When

- Data is **structured** (rows and columns)
- **Interpretability is important**
- Dataset size is **small or medium**
- Computational resources are limited

Examples:
- Healthcare diagnosis
- Credit risk analysis
- Financial prediction
- Fraud detection

In statistical ML we can explain predictions using **mathematical equations and feature importance**.

Examples of Statistical ML models:
- Linear Regression
- Logistic Regression
- Decision Trees
- Random Forest
- Support Vector Machine (SVM)

These models provide **high interpretability**.

---

## Use Deep Learning When

- Data is **complex**
- Data is **unstructured**
- Dataset size is **very large**
- High computational power is available (GPU/TPU)

Examples of unstructured data:
- Images
- Audio
- Videos
- Text
- Speech

Examples of Deep Learning applications:
- Face recognition
- Speech-to-text
- Language translation
- Chatbots
- Generative AI

Deep learning models often act like **black box models**, meaning predictions are harder to interpret.

---

# 3. Types of Neural Networks

## 1. Feedforward Neural Network (FNN)

Used mainly for **structured and static data**.

Data flows **only in one direction**.

```
Input → Hidden Layer → Hidden Layer → Output
```

Characteristics:
- No loops
- Data moves forward once

Example applications:
- Tabular prediction
- Basic classification

---

## 2. Recurrent Neural Network (RNN)

Used for **sequential data**.

RNNs contain **loops**, allowing the model to remember previous inputs.

Example applications:
- Language translation
- Speech recognition
- Chatbots
- Time series forecasting

Example idea:

```
Input → RNN → Output
        ↑
       Memory
```

RNN remembers past information.

---

## 3. Convolutional Neural Network (CNN)

Used for **image processing and computer vision**.

CNN detects patterns such as:

- edges
- textures
- shapes
- objects

Example applications:
- Image classification
- Face recognition
- Medical imaging
- Self-driving cars

---

## 4. Transformers

Modern architecture used in **Generative AI and NLP**.

Examples:
- GPT
- BERT
- T5

Transformers use **self-attention mechanisms** to understand relationships between words.

Example applications:
- ChatGPT
- Text summarization
- Machine translation
- Code generation

---

# 4. Logistic Regression as the Simplest Neural Network

Logistic Regression can be viewed as the **simplest neural network with one neuron**.

Mathematical representation:

z = (w1x1 + w2x2 + ... + wnxn) + b

Output:

y = sigmoid(z)

Diagram:

```
x1 ----\
        \
x2 ----- ( Σ ) ----> Sigmoid ----> Output
        /
x3 ----/
```

This structure represents a **single neuron**.

---

# 5. What is a Neuron?

A **neuron** is the fundamental unit of a neural network.

Steps performed by a neuron:

1. Receive inputs
2. Multiply inputs with weights
3. Add bias
4. Apply activation function
5. Produce output

Mathematical form:

```
z = Wx + b
a = activation(z)
```

---

# 6. Perceptron

A **Perceptron** is the simplest neural network model.

Structure:

```
Input → Neuron → Output
```

It works for **linearly separable problems**.

---

# 7. Multilayer Perceptron (MLP)

MLP contains **multiple hidden layers**.

Structure:

```
Input → Hidden Layer → Hidden Layer → Output
```

MLP can learn **complex nonlinear relationships**.

Both Perceptron and MLP are types of **Feedforward Neural Networks** because data flows **in one direction only**.

---

# 8. Purpose of Activation Function

Activation functions introduce **non-linearity** into neural networks.

Without activation functions, neural networks behave like **simple linear models**.

Real-world relationships are usually **non-linear**.

---

## Example: Company Bonus

A company gives bonus based on experience.

- Bonus increases with experience
- But after **30 years**, bonus stops increasing

This relationship is **non-linear**.

Activation functions help neural networks learn such patterns.

---

## Example: Score Normalization

Two interviewers evaluate a student.

Interviewer A: 7 / 10  
Interviewer B: 12 / 20  

These scores are difficult to compare.

Activation functions convert outputs to a **standard scale (0–1)**.

---

## Firing and Non-Firing Neurons

Activation functions decide whether a neuron should **activate (fire)**.

Fire → neuron sends signal forward  
Not fire → neuron output becomes zero

This helps the network select **important features**.

---

# 9. Types of Activation Functions

## 1. Sigmoid

Range:

0 to 1

Formula:

σ(x) = 1 / (1 + e^-x)

Used in:
- Binary classification
- Output layer

---

## 2. Softmax

Used for **multi-class classification**.

Example:

Cat : 0.2  
Dog : 0.6  
Bird : 0.2  

Total probability:

Sum = 1

---

## 3. Tanh

Range:

-1 to 1

Tanh is centered around zero, which helps training sometimes.

---

# 10. Problems with Sigmoid and Tanh

For very large input values:

Output becomes almost constant.

Derivative becomes very small.

This leads to:

**Vanishing Gradient Problem**

The network stops learning effectively.

---

# 11. ReLU (Rectified Linear Unit)

Formula:

ReLU(x) = max(0, x)

Advantages:
- Very fast computation
- Reduces vanishing gradient
- Widely used in deep learning

Problem:

If input is negative, output becomes zero.

If many inputs are negative → **Dead Neuron Problem**.

---

# 12. Leaky ReLU

Solution for dead neurons.

Formula:

```
x if x > 0
0.01x if x < 0
```

Allows small gradient for negative values.

---

# 13. Tensor

A **tensor** is a multidimensional array used in PyTorch.

| Dimension | Example |
|----------|--------|
| 0D | Scalar |
| 1D | Vector |
| 2D | Matrix |
| 3D | Cube |

Examples:

Scalar → 5  
Vector → [1,2,3]  
Matrix → [[1,2],[3,4]]

---

# 14. Partial Derivative

Partial derivative measures how a function changes when **one variable changes while others remain constant**.

Example:

f(x,y) = x² + y²

Partial derivatives:

∂f/∂x = 2x  
∂f/∂y = 2y

Neural networks use **partial derivatives** to update weights.

---

# 15. Autograd in PyTorch

PyTorch provides **Autograd** for automatic differentiation.

Autograd automatically computes **gradients** for tensors.

To enable gradient tracking:

```
requires_grad=True
```

---

# 16. Autograd Example

```python
import torch

x = torch.tensor(2.0, requires_grad=True)

y = x**2

y.backward()

print(x.grad)
```

Step-by-step:

Function:

y = x²

Derivative:

dy/dx = 2x

When `y.backward()` runs:

PyTorch computes derivative **2x** and evaluates it at **x = 2**.

Result:

2(2) = 4

Output:

```
tensor(4.)
```

---

# 17. NumPy vs PyTorch Tensor

| Feature | NumPy | PyTorch Tensor |
|-------|------|------|
| GPU Support | No | Yes |
| Autograd | No | Yes |
| Deep Learning Integration | No | Yes |
| Framework Integration | Limited | Fully integrated with PyTorch |

PyTorch tensors integrate directly with the **PyTorch deep learning ecosystem**.

---

# 18. torch.no_grad()

`torch.no_grad()` disables gradient computation temporarily.

Used during:
- Model inference
- Model evaluation
- Predictions

Example:

```python
with torch.no_grad():
    output = model(input)
```

Advantages:
- Saves memory
- Speeds up computation
- Prevents unnecessary gradient calculations

---

# Summary

- Deep Learning is useful for **complex unstructured data**
- Statistical ML is useful for **interpretable structured data**
- Neural networks consist of **neurons and activation functions**
- Activation functions introduce **non-linearity**
- PyTorch uses **tensors and autograd**
- Gradients are computed using **partial derivatives and backpropagation**
