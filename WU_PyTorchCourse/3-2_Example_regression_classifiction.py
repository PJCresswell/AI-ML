import torch

# Make use of a GPU or MPS (Apple) if one is available.
has_mps = torch.backends.mps.is_built()
device = "mps" if has_mps else "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.autograd import Variable

# task = 'regression'
task = 'classification'

if task == 'regression':

    # Using a NN to predict the MPG for cars
    # Read the MPG dataset.
    df = pd.read_csv("https://data.heatonresearch.com/data/t81-558/auto-mpg.csv", na_values=["NA", "?"])
    cars = df["name"]

    # Handle missing value
    df["horsepower"] = df["horsepower"].fillna(df["horsepower"].median())

    # Pandas to Numpy - training using these seven features/inputs
    x = df[["cylinders", "displacement", "horsepower", "weight", "acceleration", "year", "origin"]].values
    y = df["mpg"].values        # The actual MPG

    # Numpy to PyTorch
    x = torch.tensor(x, device=device, dtype=torch.float32)
    y = torch.tensor(y, device=device, dtype=torch.float32)

    # Create the neural network
    # Fully connected linear layers
    model = nn.Sequential(
        # Start with the seven input features going to 50 hidden neurons
        nn.Linear(x.shape[1], 50),
        # Output then goes through a rectified linear unit
        nn.ReLU(),
        # The output from these 50 hidden layer neurons then goes to 25
        nn.Linear(50, 25),
        # Rectified linear again
        nn.ReLU(),
        # Finally going to one output value - the regression value
        nn.Linear(25, 1)
    )

    # PyTorch 2.0 Model Compile (improved performance), but does not work as well on MPS
    # model = torch.compile(model,backend="aot_eager").to(device)
    model = model.to(device)

    # Define the loss function for regression
    loss_fn = nn.MSELoss()

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Train for 1000 epochs. BATCH training - complete pass
    for epoch in range(1000):
        # Zero the gradient as the beginning of the batch
        optimizer.zero_grad()
        # Flatten the output from the NN - so a vector
        out = model(x).flatten()
        # Run through loss function to see error loss across all examples in dataset
        loss = loss_fn(out, y)
        # Calculates the gradients
        loss.backward()
        # Actually modify the weights
        optimizer.step()

        # Display status every 100 epochs.
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, loss: {loss.item()}")

    pred = model(x)
    print(f"Shape: {pred.shape}")
    print(pred[0:10])

    from sklearn import metrics
    # Measure RMSE error.  RMSE is common for regression.
    score = np.sqrt(metrics.mean_squared_error(pred.cpu().detach(), y.cpu().detach()))
    print(f"Final score (RMSE): {score}")

    score = torch.sqrt(torch.nn.functional.mse_loss(pred.flatten(), y))
    print(f"Final score (RMSE): {score}")

    # Sample predictions
    for i in range(10):
        print(f"{i + 1}. Car name: {cars[i]}, MPG: {y[i]}, " + f"predicted MPG: {pred[i]}")

if task == 'classification':

    # Using a NN to predict the classes of flowers - iris dataset

    df = pd.read_csv("https://data.heatonresearch.com/data/t81-558/iris.csv", na_values=["NA", "?"])

    le = preprocessing.LabelEncoder()
    # Load in the four features
    x = df[["sepal_l", "sepal_w", "petal_l", "petal_w"]].values
    # We have three species of flowers = textual - so use index encoding to convert to 0, 1, 2
    y = le.fit_transform(df["species"])
    species = le.classes_

    x = torch.tensor(x, device=device, dtype=torch.float32)
    y = torch.tensor(y, device=device, dtype=torch.long)

    model = nn.Sequential(
        # Same architecture as before
        nn.Linear(x.shape[1], 50),
        nn.ReLU(),
        nn.Linear(50, 25),
        nn.ReLU(),
        nn.Linear(25, len(species)),
        # nn.LogSoftmax(dim=1), # Implied by use of CrossEntropyLoss
    )

    # PyTorch 2.0 Model Compile (improved performance), but does not work as well on MPS
    # model = torch.compile(model,backend="aot_eager").to(device)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()  # cross entropy loss

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for epoch in range(1000):
        optimizer.zero_grad()
        out = model(x)
        # Note: CrossEntropyLoss combines nn.LogSoftmax() and nn.NLLLoss() so don't use Softmax in the model
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, loss: {loss.item()}")

    # Print out number of species found: right from the dataset
    print(species)

    # Now trained, we can use the model
    # Tells PyTorch that finished training, want to evaluate
    model.eval()
    pred = model(x)
    print(f"Shape: {pred.shape}")
    # We get three values returned for each of the 150 examples
    print(pred[0:10])

    # Turn off scientific notation
    np.set_printoptions(suppress=True)
    print(pred[0:10].cpu().detach().numpy())

    # We want the highest prediction across the classes. Can find using argmax
    _, predict_classes = torch.max(pred, 1)
    print(f"Predictions: {predict_classes}")
    print(f"Expected: {y}")

    # Turn these indices back into species that we can understand
    print(species[predict_classes[1:10].cpu().detach()])

    # Calculate an accuracy score
    from sklearn.metrics import accuracy_score
    correct = accuracy_score(y.cpu().detach(), predict_classes.cpu().detach())
    print(f"Accuracy: {correct}")

    # Predict for a single flower
    sample_flower = torch.tensor([[5.0, 3.0, 4.0, 2.0]], device=device)
    pred = model(sample_flower)
    print(pred)
    _, predict_classes = torch.max(pred, 1)
    print(f"Predict that {sample_flower} is: {species[predict_classes]}")

    # Predict for two flowers
    sample_flower = torch.tensor(
        [[5.0, 3.0, 4.0, 2.0], [5.2, 3.5, 1.5, 0.8]], device=device
    )
    pred = model(sample_flower).to(device)
    print(pred)
    _, predict_classes = torch.max(pred, 1)
    print(f"Predict that these two flowers {sample_flower} ")
    print(f"are: {species[predict_classes.cpu().detach()]}")

    # Note : the values returned are a mix of negative and positive - not percentages adding to 1
    # Due to the log softmax function
    # Can convert to more traditional percentages using torch.exp
    print(torch.exp(pred))