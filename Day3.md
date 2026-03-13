# PyTorch Bonus Prediction Model

```python
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# Select features and target
X = df[['performance','years_of_experience','projects_completed']].values
y = df[['bonus']].values

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# Convert NumPy arrays to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Define neural network model
class BonusPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(3,1)   # 3 input features -> 1 output
        )

    def forward(self, x):
        return self.network(x)

# Create model
model = BonusPredictor()

# Define loss function
criterion = nn.MSELoss()

# Define optimizer
optimizer = optim.SGD(model.parameters(), lr=0.005)

# Training loop
epochs = 5000

for epoch in range(epochs):

    # Forward pass
    predictions = model(X_train_tensor)

    # Calculate loss
    loss = criterion(predictions, y_train_tensor)

    # Clear previous gradients
    optimizer.zero_grad()

    # Backpropagation
    loss.backward()

    # Update weights
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Evaluate model
model.eval()

with torch.no_grad():
    test_predictions = model(X_test_tensor)
    test_loss = criterion(test_predictions, y_test_tensor)

print(f"Test Loss: {test_loss}")

# Print learned parameters
for name, value in model.named_parameters():
    print(f"Name: {name}, Value: {value}")
```


# PyTorch Training Keywords – Simple Notes

**forward()**  
Defines how input passes through the neural network to produce output.

Example idea:  
Input → Neural Network Layer → Prediction

When we write:  
model(X_train_tensor)

PyTorch automatically calls the forward() function.

Simple meaning:  
forward() = how the model converts input into prediction


**loss.backward()**  
Calculates gradients (slopes) for all weights in the model.

Gradients tell the model:  
How much each weight caused the error.

This process is called Backpropagation.

Simple meaning:  
Find how weights should change to reduce error.


**optimizer.zero_grad()**  
PyTorch stores gradients from previous steps.  
If we do not clear them, they keep adding.

Example idea:  
Step 1 gradient = 2  
Step 2 gradient = 3  

Without clearing:  
Total gradient = 5

This causes incorrect learning.

So we use:  
optimizer.zero_grad()

Simple meaning:  
Clear old gradients before calculating new ones.


**optimizer.step()**  
Updates the model weights using the gradients calculated during backpropagation.

Simple meaning:  
Adjust weights to reduce the error.


**torch.no_grad()**  
Used during testing or prediction.

Example:  
with torch.no_grad():

It disables gradient calculation.

Why we use it:
- saves memory
- faster prediction
- prevents training during testing

Simple meaning:  
Prediction mode (model is not learning)


**Neural Network Training Flow**

Input Data  
↓  
forward() → model makes prediction  
↓  
Loss Function → calculates error  
↓  
backward() → finds gradients  
↓  
zero_grad() → clears old gradients  
↓  
optimizer.step() → updates weights  


**Student Learning Analogy**

forward() → student solves the question  
loss → check how wrong the answer is  
backward() → find where mistake happened  
optimizer.step() → student improves method  
zero_grad() → clear old mistakes before next question  
torch.no_grad() → exam checking only (no learning)d  
