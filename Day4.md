# Model Optimization (Easy Explanation)

Model Optimization means:
👉 Training a model in such a way that:
- It learns **faster**
- Uses **less computation**
- Gives **better predictions**

## Ways to Optimize a Model
- Different Optimizers (GD, Momentum, RMSProp, Adam)
- Regularization (L1, L2, Dropout)
- Hyperparameter Tuning

---

# Exponentially Weighted Moving Average (EWMA)

EWMA is used to **smooth values over time**.

👉 Idea:
- Recent values = more important
- Older values = less important

## Equation

v_t = β * v_{t-1} + (1 - β) * x_t

Where:
- v_t → current average
- β (beta) → decay factor (usually 0.9)
- x_t → current value

## Example Table (β = 0.9)

| Step (t) | x_t (Value) | v_t (EWMA) |
|----------|------------|------------|
| 1        | 10         | 1.0        |
| 2        | 20         | 2.9        |
| 3        | 30         | 5.61       |
| 4        | 40         | 9.05       |

👉 Observation:
- Smooth increase
- No sudden jumps

---

# Gradient Descent with Momentum

👉 Normal Gradient Descent:
- Updates weights using only current gradient
- Can be slow and zig-zag

👉 Momentum:
- Uses **past gradients also**
- Moves faster in correct direction

## Equations

v_t = β * v_{t-1} + (1 - β) * g_t  
w = w - α * v_t  

Where:
- g_t → gradient
- v_t → velocity (momentum)
- α → learning rate

## Key Points
- Reduces oscillation
- Faster convergence
- Escapes flat regions

---

# RMSProp (Root Mean Square Propagation)

👉 Idea:
- Use **average of squared gradients**
- Adjust learning rate automatically

## Equation

s_t = β * s_{t-1} + (1 - β) * g_t²  
w = w - α * (g_t / √(s_t + ε))

Where:
- s_t → squared gradient average
- ε → small number to avoid division by zero

## Key Points
- Works well for noisy data
- Reduces large updates
- Faster convergence

---

# Adam Optimizer

👉 Adam = Momentum + RMSProp

👉 It uses:
- Mean of gradients (Momentum)
- Squared gradients (RMSProp)

## Equations

m_t = β₁ * m_{t-1} + (1 - β₁) * g_t  
v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²  

w = w - α * (m_t / √(v_t + ε))

## Key Points
- Very popular optimizer
- Stable and fast
- Works well in most cases

---

# Quick Comparison

| Optimizer | Idea | Advantage |
|----------|------|----------|
| GD       | Basic gradient | Simple |
| Momentum | Uses past gradients | Faster, smoother |
| RMSProp  | Uses squared gradients | Handles noise |
| Adam     | Combines both | Best overall |

---

# Final Summary

- EWMA → Smooths values
- Momentum → Speeds up learning
- RMSProp → Adjusts learning rate
- Adam → Best combination of both

# Regularization (Easy Notes)

## What is Regularization?

Regularization is a technique used to **prevent overfitting**.

👉 Overfitting means:
- Model performs **very well on training data**
- But performs **poorly on new (unseen) data**

👉 Goal of Regularization:
- Make model **generalize better**
- Improve performance on **real-world data**

---

# Types of Regularization

- Dropout
- L1 Regularization (Lasso)
- L2 Regularization (Ridge)
- Batch Normalization
- Early Stopping

---

# Dropout Regularization

👉 Idea:
Randomly **turn off (drop)** some neurons during training

## How it works:
- In each training step, some neurons are ignored
- Model cannot rely on specific neurons
- Forces network to learn **general patterns**

## Example (PyTorch):
```python
nn.Dropout(p=0.5)
```

- p = 0.5 → 50% neurons dropped
- Common values: 0.2, 0.3, 0.5

## Benefits:
- Prevents overfitting
- Improves generalization

---

# L1 and L2 Regularization

👉 Idea:
Add **penalty to large weights** in the loss function

---

## L1 Regularization (Lasso)

- Adds penalty: sum of absolute values of weights

### Formula:
Loss = Original Loss + λ * Σ|w|

## Key Points:
- Produces **sparse models** (some weights become 0)
- Useful for **feature selection**

---

## L2 Regularization (Ridge)

- Adds penalty: sum of squares of weights

### Formula:
Loss = Original Loss + λ * Σ(w²)

## Key Points:
- Reduces weight values smoothly
- Prevents very large weights
- More commonly used than L1

---

# Batch Normalization

👉 Idea:
Normalize inputs of each layer

## What it does:
- Mean = 0
- Variance = 1

## Benefits:
1. Faster training
2. Allows higher learning rates
3. Less sensitive to initialization
4. Slight regularization effect
5. Works well with dropout

---

# Early Stopping

👉 Idea:
Stop training when model stops improving

## How it works:
- Monitor validation loss
- Stop if no improvement for some iterations

## Important Term:
- **Patience** → Number of iterations to wait before stopping

## Benefits:
- Prevents overfitting
- Saves time and computation
- Avoids unnecessary training

---

# Quick Comparison

| Technique          | Idea                          | Benefit                     |
|-------------------|-------------------------------|----------------------------|
| Dropout           | Random neurons off            | Better generalization      |
| L1 (Lasso)        | Absolute weight penalty       | Feature selection          |
| L2 (Ridge)        | Squared weight penalty       | Smooth weights             |
| Batch Norm        | Normalize inputs             | Faster + stable training   |
| Early Stopping    | Stop early                   | Saves time + avoids overfit|

---

# Final Summary

- Regularization helps **avoid overfitting**
- Makes model **robust and generalizable**
- Most commonly used:
  - Dropout
  - L2 Regularization
  - Adam + BatchNorm combination


# Deep Learning Takeaways  
## Chapter: Model Optimization – Hyperparameter Tuning

---

# What is Hyperparameter Tuning?

Hyperparameter Tuning is the process of **selecting the best values** for parameters that control how a model learns.

👉 Examples of Hyperparameters:
- Learning Rate
- Batch Size
- Number of Epochs
- Number of Layers

## Important Difference

| Type                | Description                          |
|---------------------|--------------------------------------|
| Parameters          | Learned during training (weights)     |
| Hyperparameters     | Set before training                  |

---

## Why is it Important?

1. Improves model performance  
2. Prevents overfitting and underfitting  
3. Helps model generalize better on unseen data  
4. Increases accuracy and efficiency  

---

# GridSearchCV and RandomSearchCV

---

## GridSearchCV

👉 Idea:
Try **all possible combinations** of hyperparameters

### How it works:
- Define a set of values for each hyperparameter
- Model tries every combination

### Example:
Learning Rate = [0.01, 0.1]  
Batch Size = [16, 32]  

👉 Total combinations = 4

### Pros:
- Finds the best possible combination  
- Very accurate  

### Cons:
- Very slow  
- Computationally expensive  

---

## RandomSearchCV

👉 Idea:
Try **random combinations** instead of all

### How it works:
- Randomly selects combinations from search space
- Does not try all possibilities

### Pros:
- Faster than GridSearch  
- Works well for large datasets  

### Cons:
- Might miss the absolute best combination  

---

## Key Common Point

👉 Both use **Cross-Validation**

- Split data into multiple parts  
- Train and validate multiple times  
- Reduces overfitting  

---

## Grid vs Random (Quick Comparison)

| Feature            | GridSearchCV        | RandomSearchCV       |
|-------------------|--------------------|---------------------|
| Search Type       | Exhaustive         | Random              |
| Speed             | Slow               | Fast                |
| Accuracy          | High               | Good                |
| Best For          | Small datasets     | Large datasets      |

---

# Optuna (Advanced Hyperparameter Tuning)

👉 Optuna is a modern tool for **automatic hyperparameter optimization**

---

## How it works

- Uses **trial-based search**
- Each trial = one set of hyperparameters
- Learns from previous trials

---

## Key Techniques

### 1. Bayesian Optimization
- Uses past results to choose better parameters
- Smarter than random search

### 2. Dynamic Pruning
- Stops bad trials early
- Saves time and computation

---

## Benefits of Optuna

1. Faster than Grid and Random Search  
2. Efficient for large search spaces  
3. Automatically finds optimal parameters  
4. Saves computation using pruning  

---

## When to Use Optuna?

- Deep Learning models  
- Large datasets  
- Many hyperparameters  
- Limited computational resources  

---

# Final Summary

- Hyperparameter tuning improves model performance  
- GridSearch → Accurate but slow  
- RandomSearch → Faster, less exhaustive  
- Optuna → Smart, fast, and efficient  

👉 Best Practice:
- Start with Random Search  
- Use Optuna for advanced optimization  

