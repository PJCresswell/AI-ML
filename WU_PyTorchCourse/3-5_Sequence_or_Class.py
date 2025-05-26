import torch
has_mps = torch.backends.mps.is_built()
device = "mps" if has_mps else "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

import torch.nn as nn

# Using nn.Sequential
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
    nn.LogSoftmax(dim=1)
)

# Using a class for the same example
class CustomNetwork(nn.Module):
    def __init__(self):
        super(CustomNetwork, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 10)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.log_softmax(x)
        return x

# Create an instance of the CustomNetwork
model = CustomNetwork()

# Full example using a class
import numpy as np
import pandas as pd
import torch.nn.functional as F
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.autograd import Variable


class Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Net, self).__init__()

        # Define each of the layers
        self.layer1 = nn.Linear(input_dim, 50)
        self.layer2 = nn.Linear(50, 25)
        self.layer3 = nn.Linear(25, output_dim)

    def forward(self, x):
        # Pass the input through each of the layers
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


# Load and preprocess data
df = pd.read_csv("https://data.heatonresearch.com/data/t81-558/auto-mpg.csv", na_values=["NA", "?"])

# Replace missing horsepower values with median
df["horsepower"] = df["horsepower"].fillna(df["horsepower"].median())

# Convert pandas DataFrame to PyTorch tensors
x = torch.tensor(
    df[
        [
            "cylinders",
            "displacement",
            "horsepower",
            "weight",
            "acceleration",
            "year",
            "origin",
        ]
    ].values,
    device=device,
    dtype=torch.float32,
)
y = torch.tensor(df["mpg"].values, device=device,
                 dtype=torch.float32)  # regression

# Initialize the model, loss function, and optimizer
model = Net(x.shape[1], 1).to(device)
model = torch.compile(model,backend="aot_eager").to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(1000):
    # Zero gradients
    optimizer.zero_grad()

    # Forward pass
    outputs = model(x).flatten()

    # Compute loss
    loss = loss_fn(outputs, y)

    # Backward pass and optimize
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, loss: {loss.item()}")