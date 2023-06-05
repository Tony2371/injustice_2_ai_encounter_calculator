import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn, optim
import sqlite3
from classes_and_functions import Net

conn = sqlite3.connect('injustice_2.db')
df = pd.read_sql(sql='SELECT * FROM ai_battle_log', con=conn)

# encode the character names as integers using pandas' factorize method
df['fighter_2_name'], fighter_2_index = pd.factorize(df['fighter_2_name'])
map_dict = {v: k for k, v in enumerate(fighter_2_index)}
df['fighter_1_name'] = df['fighter_1_name'].map(map_dict)

# convert the level columns to numeric format
df['fighter_1_level'] = pd.to_numeric(df['fighter_1_level'], errors='coerce')
df['fighter_2_level'] = pd.to_numeric(df['fighter_2_level'], errors='coerce')

# normalize the levels to be between 0 and 1 by dividing by 30
df['fighter_1_level'] /= 30
df['fighter_2_level'] /= 30

# encode the result as an integer (0 for lose, 1 for win)
df['result'] = df['result'].map({'loose': 0, 'win': 1})

# split the data into features and labels
X = df[['fighter_1_name', 'fighter_1_level', 'fighter_2_name', 'fighter_2_level']]
y = df['result']

# split the data into a training set and a validation set
msk = np.random.rand(len(df)) < 0.99
X_train, y_train = X[msk], y[msk]
X_val, y_val = X[~msk], y[~msk]

# convert the pandas DataFrames to PyTorch tensors
X_train = torch.tensor(X_train.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_val = torch.tensor(X_val.values, dtype=torch.float32)
y_val = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

# create PyTorch DataLoaders for the training and validation sets
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_dataset = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

# initialize the model, loss function and optimizer
model = Net()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Train the model
epochs = 100
for epoch in range(epochs):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # calculate validation loss at the end of each epoch
    model.eval()
    with torch.no_grad():
        val_loss = sum(criterion(model(inputs), labels) for inputs, labels in val_loader)
    val_loss = val_loss / len(val_loader)

    print(f"Epoch {epoch+1}/{epochs}.. "
          f"Train loss: {loss.item():.3f}.. "
          f"Validation loss: {val_loss:.3f}")
    torch.save(model.state_dict(), 'injustice_model.pth')

'''
for i, name in enumerate(fighter_2_index):
    print(i,'|', name)

while True:
    f_1 = input('Fighter 1:')
    f_1_lvl = input('Fighter 1 level:')
    f_2 = input('Fighter 2:')
    f_2_lvl = input('Fighter 2 level:')

    nn_input = torch.tensor([int(f_1), int(f_1_lvl)/30, int(f_2), int(f_2_lvl)/30])
    print(model(nn_input))
'''