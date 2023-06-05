import torch
from torch import nn, optim
import itertools

def linear_interpolation(value, input_min, input_max, output_range_min, output_range_max):
    return output_range_min + (value - input_min) * ((output_range_max - output_range_min) / (input_max - input_min))


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

class Encounter_group():
    def __init__(self, group, model, fighter_indices):
        self.encounter_1 = group[0]
        self.encounter_2 = group[1]
        self.encounter_3 = group[2]

        neural_input_1 = torch.tensor(
            [list(fighter_indices).index(self.encounter_1[0].name), self.encounter_1[0].level / 30,
             list(fighter_indices).index(self.encounter_1[1].name), self.encounter_1[1].level / 30])
        self.encounter_1_win_chance = round(model(neural_input_1).item(), 3)

        neural_input_2 = torch.tensor(
            [list(fighter_indices).index(self.encounter_2[0].name), self.encounter_2[0].level / 30,
             list(fighter_indices).index(self.encounter_2[1].name), self.encounter_2[1].level / 30])
        self.encounter_2_win_chance = round(model(neural_input_2).item(), 3)

        neural_input_3 = torch.tensor(
            [list(fighter_indices).index(self.encounter_3[0].name), self.encounter_3[0].level / 30,
             list(fighter_indices).index(self.encounter_3[1].name), self.encounter_3[1].level / 30])
        self.encounter_3_win_chance = round(model(neural_input_3).item(), 3)

        #TOTAL GROUP WIN CHANCE
        prob_win_case1 = self.encounter_1_win_chance * self.encounter_2_win_chance
        prob_win_case2 = (1 - self.encounter_1_win_chance) * self.encounter_2_win_chance * self.encounter_3_win_chance
        prob_win_case3 = self.encounter_1_win_chance * (1 - self.encounter_2_win_chance) * self.encounter_3_win_chance
        self.total_prob_win = prob_win_case1 + prob_win_case2 + prob_win_case3
        self.total_prob_win = round(self.total_prob_win, 3)



