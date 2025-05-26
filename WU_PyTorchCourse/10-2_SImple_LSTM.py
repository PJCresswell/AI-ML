# Make use of a GPU or MPS (Apple) if one is available.  (see module 3.2)
import torch
has_mps = torch.backends.mps.is_built()
device = "mps" if has_mps else "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

import torch.nn as nn
import torch.optim as optim

# Data
# One feature
# Six observations in the sequence
# Six different examples
# Is trying to predict the colour of a car as passes a camera - eg 1 is a red car ...
# Super simple exampls

max_features = 4
x_data = [
    [[0], [1], [1], [0], [0], [0]],
    [[0], [0], [0], [2], [2], [0]],
    [[0], [0], [0], [0], [3], [3]],
    [[0], [2], [2], [0], [0], [0]],
    [[0], [0], [3], [3], [0], [0]],
    [[0], [0], [0], [0], [1], [1]]
]
x = torch.tensor(x_data, dtype=torch.float32)
y = torch.tensor([1, 2, 3, 2, 3, 1], dtype=torch.int64)

# Convert labels to one-hot encoding
y2 = torch.nn.functional.one_hot(y, max_features).to(torch.float32)
print(y2)

# Model using a sequence
class LSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMLayer, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)

    def forward(self, x):
        out, _ = self.lstm(x)
        return out

model = nn.Sequential(
    LSTMLayer(input_size=1, hidden_size=128),
    nn.Dropout(p=0.2),
    nn.Flatten(),
    nn.Linear(128*6, 4),
    nn.Sigmoid()
)

# Check for GPU availability
model.to(device)
x, y2 = x.to(device), y2.to(device)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

# Train the model
print('Train...')
for epoch in range(200):
    optimizer.zero_grad()
    outputs = model(x)
    loss = criterion(outputs, y2)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/200], Loss: {loss.item():.4f}")

# Predictions
with torch.no_grad():
    outputs = model(x)
    predicted_classes = torch.argmax(outputs, 1)
    print(f"Predicted classes: {predicted_classes.cpu().numpy()}")
    print(f"Expected classes: {y.cpu().numpy()}")

# Now present a new example for classification
def runit(model, inp):
    inp = torch.tensor(inp, dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = model(inp)
    return torch.argmax(outputs[0]).item()

print(runit(model, [[[0], [2], [2], [0], [0], [0]]]))