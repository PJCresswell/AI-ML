import copy
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold

has_mps = torch.backends.mps.is_built()
device = "mps" if has_mps else "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Early stopping class (see module 3.4)
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

'''
###########################################################
# First - regression. We can just select the folds randomly

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

# Convert to PyTorch Tensors
# Looking to predict the AGE values
x_columns = df.columns.drop(['age', 'id'])
x = torch.tensor(df[x_columns].values, dtype=torch.float32, device=device)
y = torch.tensor(df['age'].values, dtype=torch.float32, device=device).view(-1, 1)

# Set random seed for reproducibility
torch.manual_seed(42)

# Cross-Validate
# Telling it to shuffle the items
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
    model = torch.compile(model,backend="aot_eager").to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    # Early Stopping variables
    best_loss = float('inf')
    early_stopping_counter = 0

    # Training loop
    EPOCHS = 500
    epoch = 0
    done = False
    es = EarlyStopping()

    while not done and epoch<EPOCHS:
        epoch += 1
        model.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(x_batch)
            loss = loss_fn(output, y_batch)
            loss.backward()
            optimizer.step()

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
'''

#########################################################
# Now classification - using stratified k-fold validation

# Read the data set
df = pd.read_csv("https://data.heatonresearch.com/data/t81-558/jh-simple-dataset.csv", na_values=["NA", "?"],)

# Generate dummies for job
df = pd.concat([df, pd.get_dummies(df["job"], prefix="job",dtype=int)], axis=1)
df.drop("job", axis=1, inplace=True)

# Generate dummies for area
df = pd.concat([df, pd.get_dummies(df["area"], prefix="area",dtype=int)], axis=1)
df.drop("area", axis=1, inplace=True)

# Missing values for income
med = df["income"].median()
df["income"] = df["income"].fillna(med)

# Standardize ranges
df["income"] = zscore(df["income"])
df["aspect"] = zscore(df["aspect"])
df["save_rate"] = zscore(df["save_rate"])
df["age"] = zscore(df["age"])
df["subscriptions"] = zscore(df["subscriptions"])

# Convert to numpy - Classification
x_columns = df.columns.drop("product").drop("id")
x = df[x_columns].values
dummies = pd.get_dummies(df["product"],dtype=int)  # Classification
products = dummies.columns
y = dummies.values

# Assuming your data is in Numpy Arrays. If not, convert them into Numpy Arrays
x = np.array(x)
y = np.array(y)

# Use the nn.Sequential API
model = nn.Sequential(
    nn.Linear(x.shape[1], 50),
    nn.ReLU(),
    nn.Linear(50, 25),
    nn.ReLU(),
    nn.Linear(25, y.shape[1]),
    nn.Softmax(dim=1),
)
model = torch.compile(model,backend="aot_eager").to(device)

# Cross-validate
kf = StratifiedKFold(5, shuffle=True, random_state=42)

# Defining Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

oos_y = []
oos_pred = []
fold = 0

for train, test in kf.split(x, df["product"]):
    fold += 1
    print(f"Fold #{fold}")

    x_train = torch.tensor(x[train], device=device, dtype=torch.float32)
    y_train = torch.tensor(np.argmax(y[train], axis=1),device=device, dtype=torch.long)  # Convert to class indices
    x_test = torch.tensor(x[test],device=device, dtype=torch.float32)
    y_test = torch.tensor(np.argmax(y[test], axis=1),device=device, dtype=torch.long)  # Convert to class indices

    # Training loop
    EPOCHS = 500
    epoch = 0
    done = False
    es = EarlyStopping(restore_best_weights=True)

    while not done and epoch < EPOCHS:
        epoch += 1
        model.train()
        optimizer.zero_grad()
        output = model(x_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

        # Evaluate validation loss
        model.eval()
        with torch.no_grad():
            y_val = model(x_test)
            val_loss = criterion(y_val, y_test)

        if es(model, val_loss):
            done = True

    # Prediction
    with torch.no_grad():
        y_val = model(x_test)
        _, pred = torch.max(y_val, 1)

    oos_y.append(y_test.cpu().numpy())
    oos_pred.append(pred.cpu().numpy())

    print(
        f"Epoch {epoch}/{EPOCHS}, Validation Loss: " f"{val_loss.item()}, {es.status}"
    )

    # Measure this fold's accuracy
    score = metrics.accuracy_score(y_test.cpu().numpy(), pred.cpu().numpy())
    print(f"Fold score (accuracy): {score}")

# Build the oos prediction list and calculate the error.
oos_y = np.concatenate(oos_y)
oos_pred = np.concatenate(oos_pred)

score = metrics.accuracy_score(oos_y, oos_pred)
print(f"Final score (accuracy): {score}")