import copy
import torch

# Make use of a GPU or MPS (Apple) if one is available.  (see module 3.2)
import torch
has_mps = torch.backends.mps.is_built()
device = "mps" if has_mps else "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Early stopping (see module 3.4)
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

import pandas as pd
from scipy.stats import zscore
from sklearn.model_selection import train_test_split

# Read the data set
df = pd.read_csv("https://data.heatonresearch.com/data/t81-558/jh-simple-dataset.csv", na_values=['NA','?'])

# Generate dummies for job
df = pd.concat([df,pd.get_dummies(df['job'],prefix="job",dtype=int)],axis=1)
df.drop('job', axis=1, inplace=True)

# Generate dummies for area
df = pd.concat([df,pd.get_dummies(df['area'],prefix="area",dtype=int)],axis=1)
df.drop('area', axis=1, inplace=True)

# Generate dummies for product
df = pd.concat([df,pd.get_dummies(df['product'],prefix="product",dtype=int)],axis=1)
df.drop('product', axis=1, inplace=True)

# Missing values for income
med = df['income'].median()
df['income'] = df['income'].fillna(med)

# Standardize ranges
df['income'] = zscore(df['income'])
df['aspect'] = zscore(df['aspect'])
df['save_rate'] = zscore(df['save_rate'])
df['subscriptions'] = zscore(df['subscriptions'])

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# Convert to PyTorch Tensors
x_columns = df.columns.drop(['age', 'id'])
x = torch.tensor(df[x_columns].values, dtype=torch.float32, device=device)
y = torch.tensor(df['age'].values, dtype=torch.float32, device=device).view(-1, 1)

# Set random seed for reproducibility
torch.manual_seed(42)

# Cross-Validate
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Early stopping parameters
patience = 10

fold = 0
for train_idx, test_idx in kf.split(x):
    fold += 1
    print(f"Fold #{fold}")

    x_train, x_test = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # PyTorch DataLoader
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Create the model and optimizer
    model = nn.Sequential(
        nn.Linear(x.shape[1], 20),
        nn.ReLU(),
        nn.Linear(20, 10),
        nn.ReLU(),
        nn.Linear(10, 1)
    )

    # Pre-compiling the model - gives a performance boost
    model = torch.compile(model, backend="aot_eager").to(device)

    # This is the key bit ! adjust learning rate every 50 epochs
    optimizer = optim.Adam(model.parameters())
    scheduler = StepLR(optimizer, step_size=50, gamma=0.90)
    loss_fn = nn.MSELoss()

    # Early Stopping variables
    best_loss = float('inf')
    early_stopping_counter = 0

    # Training loop
    EPOCHS = 500
    epoch = 0
    done = False
    es = EarlyStopping()

    while not done and epoch < EPOCHS:
        epoch += 1
        model.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(x_batch)
            loss = loss_fn(output, y_batch)
            loss.backward()
            optimizer.step()

        scheduler.step()  # apply learning rate schedule
        # Validation
        model.eval()
        with torch.no_grad():
            val_output = model(x_test)
            val_loss = loss_fn(val_output, y_test)

        if es(model, val_loss):
            done = True

    print(f"Epoch {epoch}/{EPOCHS}, Validation Loss: "
          f"{val_loss.item()}, {es.status}")

# Final evaluation
model.eval()
with torch.no_grad():
    oos_pred = model(x_test)
score = torch.sqrt(loss_fn(oos_pred, y_test)).item()
print(f"Fold score (RMSE): {score}")