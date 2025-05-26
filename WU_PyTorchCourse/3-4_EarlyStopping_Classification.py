import torch
has_mps = torch.backends.mps.is_built()
device = "mps" if has_mps else "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

import copy

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_model = None
        self.best_loss = None
        self.counter = 0
        self.status = ""

    def __call__(self, model, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model.state_dict())
        elif self.best_loss - val_loss >= self.min_delta:
            self.best_model = copy.deepcopy(model.state_dict())
            self.best_loss = val_loss
            self.counter = 0
            self.status = f"Improvement found, counter reset to {self.counter}"
        else:
            self.counter += 1
            self.status = f"No improvement in the last {self.counter} epochs"
            if self.counter >= self.patience:
                self.status = f"Early stopping triggered after {self.counter} epochs."
                if self.restore_best_weights:
                    model.load_state_dict(self.best_model)
                return True
        return False

import time

import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

def load_data():
    df = pd.read_csv("https://data.heatonresearch.com/data/t81-558/iris.csv", na_values=["NA", "?"])

    le = LabelEncoder()

    x = df[["sepal_l", "sepal_w", "petal_l", "petal_w"]].values
    y = le.fit_transform(df["species"])
    species = le.classes_

    # Split into validation and training sets
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Numpy to Torch Tensor
    x_train = torch.tensor(x_train, device=device, dtype=torch.float32)
    y_train = torch.tensor(y_train, device=device, dtype=torch.long)

    x_test = torch.tensor(x_test, device=device, dtype=torch.float32)
    y_test = torch.tensor(y_test, device=device, dtype=torch.long)

    return x_train, x_test, y_train, y_test, species


x_train, x_test, y_train, y_test, species = load_data()

# Create datasets
BATCH_SIZE = 16

dataset_train = TensorDataset(x_train, y_train)
dataloader_train = DataLoader(
    dataset_train, batch_size=BATCH_SIZE, shuffle=True)

dataset_test = TensorDataset(x_test, y_test)
dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True)

# Create model using nn.Sequential
model = nn.Sequential(
    nn.Linear(x_train.shape[1], 50),
    nn.ReLU(),
    nn.Linear(50, 25),
    nn.ReLU(),
    nn.Linear(25, len(species)),
    nn.LogSoftmax(dim=1),
)

model = torch.compile(model,backend="aot_eager").to(device)

loss_fn = nn.CrossEntropyLoss()  # cross entropy loss

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
es = EarlyStopping()

epoch = 0
done = False
while epoch < 1000 and not done:
    epoch += 1
    steps = list(enumerate(dataloader_train))
    pbar = tqdm.tqdm(steps)
    model.train()
    for i, (x_batch, y_batch) in pbar:
        y_batch_pred = model(x_batch.to(device))
        loss = loss_fn(y_batch_pred, y_batch.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss, current = loss.item(), (i + 1) * len(x_batch)
        if i == len(steps) - 1:
            model.eval()
            pred = model(x_test)
            vloss = loss_fn(pred, y_test)
            if es(model, vloss):
                done = True
            pbar.set_description(
                f"Epoch: {epoch}, tloss: {loss}, vloss: {vloss:>7f}, {es.status}"
            )
        else:
            pbar.set_description(f"Epoch: {epoch}, tloss {loss:}")

pred = model(x_test)
vloss = loss_fn(pred, y_test)
print(f"Loss = {vloss}")

from sklearn.metrics import accuracy_score

pred = model(x_test)
_, predict_classes = torch.max(pred, 1)
correct = accuracy_score(y_test.cpu(), predict_classes.cpu())
print(f"Accuracy: {correct}")