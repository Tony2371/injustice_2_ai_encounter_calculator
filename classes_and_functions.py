import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import glob
import cv2
from skimage.metrics import structural_similarity as ssim
import random

def generate_random_ai_loadout():
    numbers = [0 for i in range(6)]
    remainder = 60
    for i in range(6):
        max_value = min(remainder - (5 - i), 30)
        num = random.randint(0, max_value)
        remainder -= num
        numbers[i] = num
    numbers[-1] += remainder
    random.shuffle(numbers)

    return numbers

def fighter_one_hot(name):
    fighter_indices = {
        'aquaman': 0,
        'atom': 1,
        'atrocitus': 2,
        'bane': 3,
        'batman': 4,
        'black_adam': 5,
        'black_canary': 6,
        'black_manta': 7,
        'blue_beetle': 8,
        'brainiac': 9,
        'captain_cold': 10,
        'catwoman': 11,
        'cheetah': 12,
        'cyborg': 13,
        'darkseid': 14,
        'deadshot': 15,
        'doctor_fate': 16,
        'enchantress': 17,
        'firestorm': 18,
        'flash': 19,
        'gorilla_grodd': 20,
        'green_arrow': 21,
        'green_lantern': 22,
        'harley_quinn': 23,
        'hellboy': 24,
        'joker': 25,
        'poison_ivy': 26,
        'raiden': 27,
        'red_hood': 28,
        'robin': 29,
        'scarecrow': 30,
        'starfire': 31,
        'subzero': 32,
        'supergirl': 33,
        'superman': 34,
        'swamp_thing': 35,
        'tmnt': 36,
        'wonder_woman': 37,
        'not_selected': 38
    }

    fighter_indices.pop('not_selected')
    num_classes = len(fighter_indices)
    one_hot_dict = {key: F.one_hot(torch.tensor([value]), num_classes=num_classes) for key, value in
                    fighter_indices.items()}
    return one_hot_dict[name]


def fighter_indices(fighter_name=None, fighter_index=None):
    fighter_indices = {
        'aquaman': 0,
        'atom': 1,
        'atrocitus': 2,
        'bane': 3,
        'batman': 4,
        'black_adam': 5,
        'black_canary': 6,
        'black_manta': 7,
        'blue_beetle': 8,
        'brainiac': 9,
        'captain_cold': 10,
        'catwoman': 11,
        'cheetah': 12,
        'cyborg': 13,
        'darkseid': 14,
        'deadshot': 15,
        'doctor_fate': 16,
        'enchantress': 17,
        'firestorm': 18,
        'flash': 19,
        'gorilla_grodd': 20,
        'green_arrow': 21,
        'green_lantern': 22,
        'harley_quinn': 23,
        'hellboy': 24,
        'joker': 25,
        'poison_ivy': 26,
        'raiden': 27,
        'red_hood': 28,
        'robin': 29,
        'scarecrow': 30,
        'starfire': 31,
        'subzero': 32,
        'supergirl': 33,
        'superman': 34,
        'swamp_thing': 35,
        'tmnt': 36,
        'wonder_woman': 37,
        'not_selected': 38
    }

    if fighter_index is None:
        return fighter_indices[fighter_name]
    if fighter_name is None:
        for name, i in fighter_indices.items():
            if i == fighter_index:
                return name

def linear_interpolation(value, input_min, input_max, output_range_min, output_range_max):
    return output_range_min + (value - input_min) * ((output_range_max - output_range_min) / (input_max - input_min))

def normalize_list(lst, min_max_range = None):
    if min_max_range != None:
        min_val = min_max_range[0]
        max_val = min_max_range[1]
    else:
        min_val = min(lst)
        max_val = max(lst)
    return [(x - min_val) / (max_val - min_val) for x in lst]

def similarity(image_1, image_2):
    image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
    similarity = ssim(image_1, image_2)
    return similarity

def template_matching(image_input, template_folder, resize=None):
    probability = 0
    image_main = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
    for f in glob.glob(f'{template_folder}/*.png'):
        template = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if resize != None:
            template = cv2.resize(template, resize, interpolation=cv2.INTER_LINEAR)
        similarity = ssim(image_main, template)
        if similarity > probability:
            probability = similarity
            matched_output = f[len(template_folder)+1:].replace('.png', '')
    return matched_output, similarity

def tensorize_db_record(db_input_list, min_attr_value, max_attr_value):

    fighter_1_name = fighter_one_hot(db_input_list[0]).flatten()
    fighter_2_name = fighter_one_hot(db_input_list[1]).flatten()

    fighter_1_level = torch.tensor([db_input_list[2]/30])
    fighter_2_level = torch.tensor([db_input_list[3]/30])

    fighter_1_attributes_not_norm = [int(n.replace('[', '').replace(']', '')) for n in db_input_list[4].split(',')]
    fighter_1_attributes = torch.tensor(normalize_list(fighter_1_attributes_not_norm, min_max_range=(min_attr_value, max_attr_value)))
    fighter_2_attributes_not_norm = [int(n.replace('[', '').replace(']', '')) for n in db_input_list[5].split(',')]
    fighter_2_attributes = torch.tensor(normalize_list(fighter_2_attributes_not_norm, min_max_range=(min_attr_value, max_attr_value)))

    fighter_1_ai = torch.tensor([int(n.replace('[', '').replace(']', ''))/30 for n in db_input_list[6].split(',')])
    fighter_2_ai = torch.tensor([int(n.replace('[', '').replace(']', ''))/30 for n in db_input_list[7].split(',')])

    advantage = torch.tensor([db_input_list[8]])


    output = torch.cat([fighter_1_name, fighter_2_name, fighter_1_level, fighter_2_level, fighter_1_attributes, fighter_2_attributes, fighter_1_ai, fighter_2_ai, advantage])

    return output.flatten()

def tensorize_db_record_inverted(db_input_list, min_attr_value, max_attr_value):

    fighter_2_name = fighter_one_hot(db_input_list[0]).flatten()
    fighter_1_name = fighter_one_hot(db_input_list[1]).flatten()

    fighter_2_level = torch.tensor([db_input_list[2] / 30])
    fighter_1_level = torch.tensor([db_input_list[3] / 30])

    fighter_2_attributes_not_norm = [int(n.replace('[', '').replace(']', '')) for n in db_input_list[4].split(',')]
    fighter_2_attributes = torch.tensor(
        normalize_list(fighter_2_attributes_not_norm, min_max_range=(min_attr_value, max_attr_value)))
    fighter_1_attributes_not_norm = [int(n.replace('[', '').replace(']', '')) for n in db_input_list[5].split(',')]
    fighter_1_attributes = torch.tensor(
        normalize_list(fighter_1_attributes_not_norm, min_max_range=(min_attr_value, max_attr_value)))

    fighter_2_ai = torch.tensor(
        [int(n.replace('[', '').replace(']', '')) / 30 for n in db_input_list[6].split(',')])
    fighter_1_ai = torch.tensor(
        [int(n.replace('[', '').replace(']', '')) / 30 for n in db_input_list[7].split(',')])

    advantage = torch.tensor([db_input_list[8] * -1])

    output = torch.cat(
        [fighter_1_name, fighter_2_name, fighter_1_level, fighter_2_level, fighter_1_attributes, fighter_2_attributes,
         fighter_1_ai, fighter_2_ai, advantage])

    return output.flatten()

class ModelDigitRecognition(nn.Module):
    def __init__(self):
        super(ModelDigitRecognition, self).__init__()
        self.convolution = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3)
        self.fc1 = nn.Linear(1440, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.mish = nn.Mish()

    def forward(self, x):
        x = x.float()
        x = x.permute(0, 3, 1, 2)
        x = torch.relu(self.convolution(x))
        x = x.contiguous().view(x.size(0), -1)
        x = self.mish(self.fc1(x))
        x = self.mish(self.fc2(x))
        x = self.fc3(x)
        return x

class ModelHpTrack(nn.Module):
    def __init__(self):
        super(ModelHpTrack, self).__init__()

        self.conv_layer_1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3)
        self.conv_layer_2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)
        self.pooling_layer = nn.MaxPool2d(2, 2)
        self.mish = nn.Mish()
        self.dropout = nn.Dropout(0.25)

        self.fc1 = nn.Linear(53504, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc_output = nn.Linear(512, 1)  # 1 output for regression problem

    def forward(self, x):
        x = self.conv_layer_1(x)
        x = torch.relu(x)  # Apply activation
        x = self.pooling_layer(x)  # Apply pooling

        # Apply second convolutional layer
        x = self.conv_layer_2(x)
        x = torch.relu(x)  # Apply activation
        x = self.pooling_layer(x)  # Apply pooling

        # Flatten the output from the conv layers to fit into the FC layers
        x = x.view(x.size(0), -1)

        # Apply FC layers with activation
        x = self.mish(self.fc1(x))
        x = self.dropout(x)
        x = self.mish(self.fc2(x))
        x = self.dropout(x)
        x = self.mish(self.fc3(x))
        x = self.fc_output(x)  # Apply fc3
        return x

class ModelFighterRecognition(nn.Module):
    def __init__(self):
        super(ModelFighterRecognition, self).__init__()

        # Fully connected layers
        self.fc1 = nn.Linear(19200, 512) #19200 IS 80x80 images with 3 channels
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 39)
        self.mish = nn.Mish()

    def forward(self, x):
        #x = x.view(x.size(0), -1)
        x = x.flatten()
        x = x.float()
        x = self.mish(self.fc1(x))
        x = self.mish(self.fc2(x))
        x = self.fc3(x)  # Apply fc3
        return x

class ModelAdvantage(nn.Module):
    def __init__(self):
        super(ModelAdvantage, self).__init__()

        # Fully connected layers
        self.fc1 = nn.Linear(98, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.mish = nn.Mish()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = x.float()
        x = self.mish(self.fc1(x))
        x = self.dropout(x)
        x = self.mish(self.fc2(x))
        x = self.dropout(x)
        x = torch.tanh(self.fc3(x))
        return x

class Row():
    def __init__(self, index):
        self.characters = [None, None, None]
        self.levels = [None, None, None]

        self.index = index

class Fighter():
    def __init__(self, name, level, ai_primary=None, ai_secondary=None, attributes=None):
        self.name = name
        self.level = level
        self.ai_primary = ai_primary
        self.ai_secondary = ai_secondary
        self.selected_ai = 'primary'
        self.attributes = attributes

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



