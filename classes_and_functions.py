import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import glob
import cv2
from skimage.metrics import structural_similarity as ssim
import random
from colorama import Fore, Style

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

def abilities_indices(ability_name=None, ability_index=None):
    abilities_dict = {'calloformmarius': 0, 'fromthedeep': 1, 'ormsmariuscharge': 2, 'powerofneptune': 3, 'tidalwave': 4, 'tridentstrike': 5, 'airatomizer': 6, 'atomfield': 7, 'massiveslam': 8, 'bloodpush': 9, 'causticpool': 10, 'dexstarsrage': 11, 'pillarofblood': 12, 'siphonpower': 13, 'airtoprope': 14, 'bodypress': 15, 'doublepunch': 16, 'dropkick': 17, 'juggernautrush': 18, 'advancedparry': 19, 'airdownwardbatarang': 20, 'airstraightbatarang': 21, 'batstrike': 22, 'boomerangbat': 23, 'capeparry': 24, 'dualbatarangs': 25, 'evadingbat': 26, 'taserrangs': 27, 'airlightningcage': 28, 'deitysbolt': 29, 'powerofaton': 30, 'rollingthunder': 31, 'sethstrike': 32, 'soulofshazam': 33, 'aircanarycry': 34, 'bodybounce': 35, 'clutchcrush': 36, 'risingkick': 37, 'screech': 38, 'sonicburst': 39, 'darksuit': 40, 'riptide': 41, 'sweepingmantarays': 42, 'aculeusstrike': 43, 'airdivinitymandiblestrike': 44, 'aliencloak': 45, 'mandibleflurry': 46, 'myrules': 47, 'destructionorb': 48, 'enchancedlivingmetal': 49, 'ioncannon': 50, 'tendrilharpoon': 51, 'tendrilslide': 52, 'aircoldblast': 53, 'airicebridge': 54, 'airiceout': 55, 'coldstream': 56, 'iceburst': 57, 'icerain': 58, 'shoulderofcold': 59, 'upwardcoldblast': 60, '9lives': 61, 'aircatslash': 62, 'catcall': 63, 'catclaws': 64, 'chaoticcat': 65, 'lowwhip': 66, 'aircheetahclutch': 67, 'airjunglejump': 68, 'creepingpredator': 69, 'savageslam': 70, 'spottedtorpedo': 71, 'bodyshield': 72, 'directedarmblaster': 73, 'airboomtubeaway': 74, 'boomrubeaway': 75, 'boomtube': 76, 'hateslam': 77, 'maximumomega': 78, 'reveresomegabeams': 79, 'airsniper': 80, 'heavyartillery': 81, 'lowwristcannon': 82, 'rocketjump': 83, 'scopedshot': 84, 'upshot': 85, 'absorptionspell': 86, 'airamonrablast': 87, 'airdash': 88, 'darkankh': 89, 'instantjudgement': 90, 'powerless': 91, 'seekingdisplacerorb': 92, 'anotherdimension': 93, 'banishingblast': 94, 'barrierspell': 95, 'demonsdissolve': 96, 'divinityspell': 97, 'hypnoticspell': 98, 'airgroundspark': 99, 'atombomb': 100, 'flamephase': 101, 'meltingpoint': 102, 'vaporize': 103, 'airspinout': 104, 'fistsfrenzy': 105, 'lightningcharge': 106, 'sonicbolt': 107, 'sonicparry': 108, 'speednado': 109, 'doubledribble': 110, 'furiousflurry': 111, 'gorillagrab': 112, 'primitivepound': 113, 'bolaarrow': 114, 'distractingshot': 115, 'evasiveshock': 116, 'skybolt': 117, 'smokearrow': 118, 'airoasrocket': 119, 'lanternsbarrier': 120, 'minigun': 121, 'oasrocket': 122, 'quickcharge': 123, 'rocketpower': 124, 'turbinesmash': 125, 'allpurposefrosting': 126, 'cherrybomb': 127, 'confetticannon': 128, 'helovesme': 129, 'ivysblessing': 130, 'jokerinfection': 131, 'mollywhop': 132, 'ticktock': 133, 'airdevilsrevolver': 134, 'azzaelsguard': 135, 'gravedigger': 136, 'hellsfury': 137, 'crowbarcrush': 138, 'crowbarfling': 139, 'gasser': 140, 'sideorderofpie': 141, 'surprise': 142, 'crawlingvines': 143, 'deadlythorns': 144, 'flowingearth': 145, 'petalsoflife': 146, 'thistlecoat': 147, 'airsparkport': 148, 'electriccurrent': 149, 'electricstorm': 150, 'powerbolt': 151, 'statictraps': 152, 'akimboblaze': 153, 'gutted': 154, 'hiddenexplosive': 155, 'instantdeath': 156, 'shrapnelblast': 157, 'timebomb': 158, 'airdeadlydirdarang': 159, 'deadlybirdarang': 160, 'elusiveswoop': 161, 'lineinthesand': 162, 'lowsmartbirdarang': 163, 'staffofgrayson': 164, 'floorflame': 165, 'panicportaway': 166, 'sacrifice': 167, 'soaringmurder': 168, 'terrorcharge': 169, 'novaburst': 170, 'spincycle': 171, 'starblastbarrage': 172, 'starslam': 173, 'xhalstrength': 174, 'airclonekick': 175, 'barrieroffrost': 176, 'groundfreeze': 177, 'hammerslam': 178, 'iceport': 179, 'klonecharge': 180, 'airheatvision': 181, 'earthshatter': 182, 'kryptoniangrinder': 183, 'speedingbullet': 184, 'suncharge': 185, 'airheatzap': 186, 'empoweredheatzap': 187, 'groundtremor': 188, 'kryptoncharge': 189, 'meteordrop': 190, 'truckpunchpull': 191, 'bonsai': 192, 'sinkingslough': 193, 'swampjuice': 194, 'pizzaparty': 195, 'raisingshell': 196, 'shellslide': 197, 'aegisofzeus': 198, 'airamazonianslam': 199, 'amaltheasprotection': 200, 'artemisstrength': 201, 'athenaspower': 202, 'demetersspirit': 203, 'hermesblessing': 204, 'hestiasgift': 205, 'lassospin': 206, 'swordofathena': 207, 'ability': 208, 'allpowerful': 209, 'almightypower': 210, 'augment': 211, 'boosted': 212, 'buddysystem': 213, 'deadlytransition': 214, 'eternallife': 215, 'feelthepain': 216, 'feelthepower': 217, 'gettingstronger': 218, 'godlike': 219, 'holdingback': 220, 'juggernaut': 221, 'liveforever': 222, 'longlife': 223, 'notdeadyet': 224, 'nothankyou': 225, 'notsostrong': 226, 'pumpedup': 227, 'steamroller': 228, 'strongerfaster': 229, 'tank': 230, 'truechampion': 231, 'whatiwant': 232}

    abilities_dict['not_available'] = max(abilities_dict.values())+1
    if ability_index is None:
        return abilities_dict[ability_name]
    if ability_name is None:
        for name, i in abilities_dict.items():
            if i == ability_index:
                return name

def ability_recognize(input_image, model):

    image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    if image is None:
        print("Couldn't load the image. Make sure it is an image file.")
        return

    # Preprocess the image
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    image = transform(image)
    image = image.unsqueeze(0)

    # Predict the class of the image
    with torch.no_grad():
        model.eval()
        output = model(image)
        _, predicted_class = torch.max(output, 1)

    # Calculate softmax for the output vector
    sm = torch.nn.Softmax(dim=1)
    probabilities = sm(output)

    return predicted_class.item(), probabilities[0][predicted_class.item()].item()


def tensorize_db_record_abilities(db_input_list, min_attr_value, max_attr_value):

    fighter_1_name = torch.tensor([fighter_indices(fighter_name=db_input_list[0])])
    fighter_2_name = torch.tensor([fighter_indices(fighter_name=db_input_list[1])])

    fighter_1_level = torch.tensor([db_input_list[2]/30])
    fighter_2_level = torch.tensor([db_input_list[3]/30])

    fighter_1_attributes_not_norm = [int(n.replace('[', '').replace(']', '')) for n in db_input_list[4].split(',')]
    fighter_1_attributes = torch.tensor(normalize_list(fighter_1_attributes_not_norm, min_max_range=(min_attr_value, max_attr_value)))
    fighter_2_attributes_not_norm = [int(n.replace('[', '').replace(']', '')) for n in db_input_list[5].split(',')]
    fighter_2_attributes = torch.tensor(normalize_list(fighter_2_attributes_not_norm, min_max_range=(min_attr_value, max_attr_value)))

    fighter_1_ai = torch.tensor([int(n.replace('[', '').replace(']', ''))/30 for n in db_input_list[6].split(',')])
    fighter_2_ai = torch.tensor([int(n.replace('[', '').replace(']', ''))/30 for n in db_input_list[7].split(',')])

    advantage = torch.tensor([db_input_list[8]])

    if db_input_list[9] > 0:
        fighter_1_ability_1 = torch.tensor([abilities_indices(ability_name=db_input_list[10])])
        fighter_1_ability_2 = torch.tensor([abilities_indices(ability_name=db_input_list[11])])
        fighter_1_augment = torch.tensor([abilities_indices(ability_name=db_input_list[12])])

        fighter_2_ability_1 = torch.tensor([abilities_indices(ability_name=db_input_list[13])])
        fighter_2_ability_2 = torch.tensor([abilities_indices(ability_name=db_input_list[14])])
        fighter_2_augment = torch.tensor([abilities_indices(ability_name=db_input_list[15])])


    else:
        fighter_1_ability_1 = torch.tensor([abilities_indices(ability_name='not_available')])
        fighter_1_ability_2 = torch.tensor([abilities_indices(ability_name='not_available')])
        fighter_1_augment = torch.tensor([abilities_indices(ability_name='not_available')])

        fighter_2_ability_1 = torch.tensor([abilities_indices(ability_name='not_available')])
        fighter_2_ability_2 = torch.tensor([abilities_indices(ability_name='not_available')])
        fighter_2_augment = torch.tensor([abilities_indices(ability_name='not_available')])

    output = torch.cat([fighter_1_name, fighter_2_name, fighter_1_ability_1, fighter_1_ability_2, fighter_1_augment, fighter_2_ability_1, fighter_2_ability_2, fighter_2_augment, fighter_1_level, fighter_2_level, fighter_1_attributes, fighter_2_attributes, fighter_1_ai, fighter_2_ai, advantage])

    return output.flatten()


def tensorize_db_record_abilities_inverted(db_input_list, min_attr_value, max_attr_value):

    fighter_1_name = torch.tensor([fighter_indices(fighter_name=db_input_list[1])])
    fighter_2_name = torch.tensor([fighter_indices(fighter_name=db_input_list[0])])

    fighter_1_level = torch.tensor([db_input_list[3]/30])
    fighter_2_level = torch.tensor([db_input_list[2]/30])

    fighter_1_attributes_not_norm = [int(n.replace('[', '').replace(']', '')) for n in db_input_list[5].split(',')]
    fighter_1_attributes = torch.tensor(normalize_list(fighter_1_attributes_not_norm, min_max_range=(min_attr_value, max_attr_value)))
    fighter_2_attributes_not_norm = [int(n.replace('[', '').replace(']', '')) for n in db_input_list[4].split(',')]
    fighter_2_attributes = torch.tensor(normalize_list(fighter_2_attributes_not_norm, min_max_range=(min_attr_value, max_attr_value)))

    fighter_1_ai = torch.tensor([int(n.replace('[', '').replace(']', ''))/30 for n in db_input_list[7].split(',')])
    fighter_2_ai = torch.tensor([int(n.replace('[', '').replace(']', ''))/30 for n in db_input_list[6].split(',')])

    advantage = -torch.tensor([db_input_list[8]])


    if db_input_list[9] > 0:
        fighter_1_ability_1 = torch.tensor([abilities_indices(ability_name=db_input_list[13])])
        fighter_1_ability_2 = torch.tensor([abilities_indices(ability_name=db_input_list[14])])
        fighter_1_augment = torch.tensor([abilities_indices(ability_name=db_input_list[15])])

        fighter_2_ability_1 = torch.tensor([abilities_indices(ability_name=db_input_list[10])])
        fighter_2_ability_2 = torch.tensor([abilities_indices(ability_name=db_input_list[11])])
        fighter_2_augment = torch.tensor([abilities_indices(ability_name=db_input_list[12])])
    else:
        fighter_1_ability_1 = torch.tensor([abilities_indices(ability_name='not_available')])
        fighter_1_ability_2 = torch.tensor([abilities_indices(ability_name='not_available')])
        fighter_1_augment = torch.tensor([abilities_indices(ability_name='not_available')])

        fighter_2_ability_1 = torch.tensor([abilities_indices(ability_name='not_available')])
        fighter_2_ability_2 = torch.tensor([abilities_indices(ability_name='not_available')])
        fighter_2_augment = torch.tensor([abilities_indices(ability_name='not_available')])

    output = torch.cat([fighter_1_name, fighter_2_name, fighter_1_ability_1, fighter_1_ability_2, fighter_1_augment, fighter_2_ability_1, fighter_2_ability_2, fighter_2_augment, fighter_1_level, fighter_2_level, fighter_1_attributes, fighter_2_attributes, fighter_1_ai, fighter_2_ai, advantage])
    return output.flatten()

def score_encounter_predictions(p_raw):
    #p = normalize_list(advantage_list, min_max_range=(-1, 1))
    # check if the length of the list is 3

    p = [a if a > 0 else 0 for a in p_raw]

    if len(p) != 3:
        raise ValueError("The list must contain exactly three probabilities.")

    # check if probabilities are valid (between 0 and 1)
    for prob in p:
        if prob < 0 or prob > 1:
            raise ValueError("All probabilities must be between 0 and 1.")

    p1_2 = p[0] * p[1]
    # first and third event are successful, second event fails
    p1_3 = p[0] * (1 - p[1]) * p[2]
    # first event fails, second and third event are successful
    p2_3 = (1 - p[0]) * p[1] * p[2]

    # sum of probabilities
    total_p = p1_2 + p1_3 + p2_3

    return round(total_p, 3)

def print_prediction(input_dict):
    vs_1_1 = str(input_dict["player_fighter_1_name"])+" "+str(input_dict["player_fighter_1_level"])+" "+str(input_dict["player_ai_1"])
    vs_1_2 = str(input_dict["opponent_fighter_1_name"])+" "+str(input_dict["opponent_fighter_1_level"])
    advantage_1 = round(input_dict["advantage_1"], 3)
    if len(vs_1_1) < 25:
        vs_1_1 += ' '*(25-len(vs_1_1))
    if len(vs_1_2) < 15:
        vs_1_2 += ' '*(15-len(vs_1_2))

    vs_2_1 = str(input_dict["player_fighter_2_name"]) + " " + str(input_dict["player_fighter_2_level"]) + " " + str(
        input_dict["player_ai_2"])
    vs_2_2 = str(input_dict["opponent_fighter_2_name"]) + " " + str(input_dict["opponent_fighter_2_level"])
    advantage_2 = round(input_dict["advantage_2"], 3)
    if len(vs_2_1) < 25:
        vs_2_1 += ' ' * (25 - len(vs_2_1))
    if len(vs_2_2) < 15:
        vs_2_2 += ' ' * (15 - len(vs_2_2))

    vs_3_1 = str(input_dict["player_fighter_3_name"]) + " " + str(input_dict["player_fighter_3_level"]) + " " + str(
        input_dict["player_ai_1"])
    vs_3_2 = str(input_dict["opponent_fighter_3_name"]) + " " + str(input_dict["opponent_fighter_3_level"])
    advantage_3 = round(input_dict["advantage_3"], 3)
    if len(vs_3_1) < 25:
        vs_3_1 += ' ' * (25 - len(vs_3_1))
    if len(vs_3_2) < 15:
        vs_3_2 += ' ' * (15 - len(vs_3_2))


    print(Fore.LIGHTBLUE_EX+f'Total win chance: {round(input_dict["score"]*100, 1)} %'+Style.RESET_ALL)
    print(vs_1_1 + ' vs.   ' + vs_1_2 + "  |  " + f'Win chance: {round(advantage_1*100,1)} %')
    print(vs_2_1 + ' vs.   ' + vs_2_2 + "  |  " + f'Win chance: {round(advantage_2*100,1)} %')
    print(vs_3_1 + ' vs.   ' + vs_3_2 + "  |  " + f'Win chance: {round(advantage_3*100,1)} %')
    print(Fore.LIGHTGREEN_EX+'-'*10+Style.RESET_ALL)

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

class ModelAbilityRecognition(nn.Module):
    def __init__(self, num_classes):
        super(ModelAbilityRecognition, self).__init__()
        self.convolution = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1) # INPUT IS GRAYSCALE IMAGE
        self.pool = nn.MaxPool2d(2, 2) # Add a pooling layer
        self.fc1 = nn.Linear(180*20*16, 512) # Change the input size to reflect the pooling layer
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, num_classes) # Use num_classes here
        self.mish = nn.Mish()

    def forward(self, x):
        x = x.float()
        x = self.pool(torch.relu(self.convolution(x))) # Apply the pooling layer after the ReLU activation
        x = x.contiguous().view(x.size(0), -1)
        x = self.mish(self.fc1(x))
        x = self.mish(self.fc2(x))
        x = self.fc3(x)
        return x


class ModelAdvantage_v2(nn.Module):
    def __init__(self):
        super(ModelAdvantage_v2, self).__init__()

        self.num_abilties = 233
        # Embedding layers
        self.fighter_name = nn.Embedding(38, 100)
        self.ability_embedding = nn.Embedding(self.num_abilties+1, 100)

        # Fully connected layers
        self.fc1 = nn.Linear(822, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc_out = nn.Linear(256, 1)

        self.mish = nn.Mish()
        self.dropout = nn.Dropout(0.15)

    def forward(self, x):
        # Get embeddings
        fighter1_embed = self.fighter_name(x[:, 0].long())
        fighter2_embed = self.fighter_name(x[:, 1].long())
        fighter_1_ability_1_embed = self.ability_embedding(x[:, 2].long())
        fighter_1_ability_2_embed = self.ability_embedding(x[:, 3].long())
        fighter_1_augment_embed = self.ability_embedding(x[:, 4].long())
        fighter_2_ability_1_embed = self.ability_embedding(x[:, 5].long())
        fighter_2_ability_2_embed = self.ability_embedding(x[:, 6].long())
        fighter_2_augment_embed = self.ability_embedding(x[:, 7].long())

        # Concatenate embeddings
        x = torch.cat((fighter1_embed, fighter2_embed,
                       fighter_1_ability_1_embed, fighter_1_ability_2_embed, fighter_1_augment_embed,
                       fighter_2_ability_1_embed, fighter_2_ability_2_embed, fighter_2_augment_embed,
                       x[:, 8:]), dim=1)

        # Feed through network
        x = x.view(x.size(0), -1)
        x = x.float()
        x = self.mish(self.fc1(x))
        x = self.dropout(x)
        x = self.mish(self.fc2(x))
        x = self.dropout(x)
        x = self.mish(self.fc3(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc_out(x))

        return x

class Row():
    def __init__(self, index):
        self.characters = [None, None, None]
        self.levels = [None, None, None]

        self.index = index


class Fighter():
    def __init__(self, name, level, ai_primary=None, ai_secondary=None, attributes=None, ability_1=None, ability_2=None, augment=None):
        self.name = name
        self.level = level
        self.ai_primary = ai_primary
        self.ai_secondary = ai_secondary
        self.selected_ai = 'primary'
        self.attributes = attributes
        self.ability_1 = ability_1
        self.ability_2 = ability_2
        self.augment = augment


class ModelAdvantage_testng(nn.Module):
    def __init__(self):
        super(ModelAdvantage_testng, self).__init__()

        self.num_abilties = 233
        # Embedding layers
        self.fighter_name = nn.Embedding(38, 152)
        self.ability_embedding = nn.Embedding(self.num_abilties+1, 100)

        # Fully connected layers
        self.fc1 = nn.Linear(774, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc_out = nn.Linear(512, 1)

        self.mish = nn.Mish()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # Get embeddings
        fighter2_embed = self.fighter_name(x[:, 1].long())
        fighter_1_ability_1_embed = self.ability_embedding(x[:, 2].long())
        fighter_1_ability_2_embed = self.ability_embedding(x[:, 3].long())
        fighter_1_augment_embed = self.ability_embedding(x[:, 4].long())
        fighter_2_ability_1_embed = self.ability_embedding(x[:, 5].long())
        fighter_2_ability_2_embed = self.ability_embedding(x[:, 6].long())
        fighter_2_augment_embed = self.ability_embedding(x[:, 7].long())

        # Concatenate embeddings
        x = torch.cat((fighter2_embed,
                       fighter_1_ability_1_embed, fighter_1_ability_2_embed, fighter_1_augment_embed,
                       fighter_2_ability_1_embed, fighter_2_ability_2_embed, fighter_2_augment_embed,
                       x[:, 8:]), dim=1)

        # Feed through network
        x = x.view(x.size(0), -1)
        x = x.float()
        x = self.mish(self.fc1(x))
        x = self.dropout(x)
        x = self.mish(self.fc3(x))
        x = self.dropout(x)
        x = self.mish(self.fc4(x))
        x = self.dropout(x)
        x = torch.tanh(self.fc_out(x))

        return x