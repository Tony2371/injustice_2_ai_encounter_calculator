import torch
from torch import nn, optim

def calculate_total_probability_ordered(probabilities):
    # Case 1: first two events are successful
    p1 = probabilities[0] * probabilities[1]

    # Case 2: first and third events are successful, second event fails
    p2 = probabilities[0] * (1 - probabilities[1]) * probabilities[2]

    # Case 3: first event fails, second and third events are successful
    p3 = (1 - probabilities[0]) * probabilities[1] * probabilities[2]

    # Total probability is the sum of the individual probabilities
    total_probability = p1 + p2 + p3

    return total_probability

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))

class Row():
    def __init__(self, image, index):
        self.characters = [None, None, None]
        self.levels = [None, None, None]
        self.image = image
        self.index = index

class Fighter():
    def __init__(self, name, level):
        self.name = name
        self.level = level