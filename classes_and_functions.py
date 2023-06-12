import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import glob
import cv2
from skimage.metrics import structural_similarity as ssim
import random

# LOW THRESHOLD FOR WHITE IS (245, 245, 245)
fighters_hash_pixels = {
    'aquaman': 'c35bd06599649911c27ada464cd0670649564100cb95d78223d5193e8949216c',
    'atrocitus': 'e048b321a45473471d19f2668842b5f10d90342d563f8c3d9820b2c60d095319',
    'atom': '868fbf3c7c0d3ea00f11dd8a5a2b00632b8588a4f7259d8ba648a19ffeeba7e2',
    'bane': '3ae774e81e12f712cc6688e1beaf800f5675066f9265f679be3ea1a5edb94f58',
    'batman': '6b3b1ebf1883408c2115488aa28d8ec6232b0fc3e6086813bc26c77a0a225e6f',
    'black_adam': 'dac793e750c1ada759d6de383bf7e35ba88044059c25c7a39b9149a7f52d0ad7',
    'black_canary': '9c6e498484c58dbd76fd5895f3b48a2d8b9f400d8a8466426dff2bdea1f3ee81',
    'black_manta': 'd0a78ac431960eb3566809e852eddd3b23d5e3192247947f150d4dec460024ae',
    'blue_beetle': '3d3c70da2949bd2bdba40a0b3c4e5d5dbc15afaab8573b39aa1b036fd3dcdd32',
    'brainiac': '00f23bf5847443d54c5c6fd41a4933f18d4ee0ccddb2250db4be0fe4dd61e3c9',
    'captain_cold': '997c37ae20fe7600e24dcf2b073131c2611ad022e78f810d662130ec45faae32',
    'catwoman': '25542b8678add257464183322b7bb6b9ed690041079afa8a36c56835c83631cb',
    'cheetah': '0e5b15a5afc10f087848fe1c76cdca4702193f33a712a7c334391d0ae6d7c4d3',
    'cyborg': 'ab5670e43632a01f6ca496170aa3b1ec856099527c9e4c9afca6fcbc7abdf7c1',
    'darkseid': '1808ba9eed5bfe6eddf6f9505bf95bd97d01436b83dac07474ba427967a67200',
    'deadshot': '2f07d0640e5d78826a85768fca4737c17a598e1464aac833306113cd6506faa1',
    'doctor_fate': '511d975fbcd0527e7d4705bcc840fd8f7e7fcd48811ae6991aff4f02dc6e2b2e',
    'enchantress': '3a5bee3fe501cf5e66127b7ed40c4c7fb60c53aa009d650d72ed16ac0aafe8aa',
    'firestorm': '91ef72bd9b243c6152fbc29409001269d1cc5d8a5ed2bc3619dd8e98ef51132c',
    'flash': '9d084437ff30d9b1bd01da0037d319f7edbb1bf1de7b1d5616604d01f7ed3efe',
    'gorilla_grodd': 'd84fb04a5f7475c0d7b2c25df0543697b9c16b241a53e6593c203b86b7f42caf',
    'green_arrow': '249e439a6c9ec33ed42ff872c7144858009687411bbe9c18a156515cc677c884',
    'green_lantern': '9028e48eae45342cfe197342de2e40bb534b5256c8e57b5dff58266e8e3fbb73',
    'harley_quinn': '07628decfb4ef8b3161f2ddcc12fb20d850b3202452dda301e7528fb7ad53e70',
    'hellboy': '16a6ed66d93dbb1f11216b96fe1476689bf9a80826d8b786a5bd06c1084ff7cc',
    'joker': '36a0905182a9aa3dc9533c528d0cd260d442d1f13ca3a12d8a3c2b36d3dc6b97',
    'poison_ivy': '84bad32d0c548e6319a60528432d061cf0ae8ca1bb004f56dcdf9ee8cdc27ff8',
    'raiden': 'b290f16fad5547fb45b079ed85ae748dacab924d4ffdb4f2e95ba8c6dcc47479',
    'red_hood': '31bf0cf838319b31658c85d2bbcc3faaacaa98538ddb51ddbecd654f180e3238',
    'robin': '75ff47238d43f56ef051396886fdd5b5ad3bdc05a62a9b662b3f3b2a5e4daf8e',
    'scarecrow': '2e626accec7cc2dbffdaf8e76c721c281e09ec7cf2e9b18e8842bed5742c0dec',
    'starfire': '10ce764e1a3776f9b1d4c53c2d1b390f226b1d12ba2e490e80f89605f4aba12f',
    'sub_zero': '717c1ab43d988d4ef28d6388bc7491c09e2c72d6b98e16a2a647586bdec2655c',
    'supergirl': 'cfa2e02ad63d625198ef1cb73ca3389e3aaabf3de161db6da0cb66591a1f5ec3',
    'superman': '20bcc715bbbcf25590a41a2a2dd39050cdb03edf46fbe33714a846759104c583',
    'swamp_thing': '640ac940fd3e2a53ac9e41be6a65d2ec27dbeeddcf83d358af29b4125d31edeb',
    'tmnt': '39c87d17e4620909ffc740fd685fca085dac4f332842e0d4de4098ed6481f414',
    'wonder_woman': '385683730cfe3b7f50005d14ba1ff03d68f61dfd10d39f4fb38c7cf34a58d24e',
}

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

def flattened_masked_image(image, mask_path):
    pixel_indices = np.nonzero(cv2.imread(mask_path, 0).flatten())[0]
    blue_channel = image[:, :, 0].flatten()[pixel_indices]
    green_channel = image[:, :, 1].flatten()[pixel_indices]
    red_channel = image[:, :, 2].flatten()[pixel_indices]
    normalized_image = np.concatenate((blue_channel / 255, green_channel / 255, red_channel / 255))
    return normalized_image


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

        self.fc1 = nn.Linear(53504, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 1)  # 1 output for regression problem

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
        x = self.mish(self.fc2(x))
        x = self.fc3(x)  # Apply fc3
        return x

class ModelFighterRecognition(nn.Module):
    def __init__(self):
        super(ModelFighterRecognition, self).__init__()

        # Fully connected layers
        self.fc1 = nn.Linear(19200, 512) #19200 IS 80x80 images with 3 channels
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 39)  # 1 output for regression problem
        self.mish = nn.Mish()

    def forward(self, x):
        #x = x.view(x.size(0), -1)
        x = x.flatten()
        x = x.float()
        x = self.mish(self.fc1(x))
        x = self.mish(self.fc2(x))
        x = self.fc3(x)  # Apply fc3
        return x


class Row():
    def __init__(self, image, index):
        self.characters = [None, None, None]
        self.levels = [None, None, None]
        self.image = image
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



