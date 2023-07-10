import torch
from torch import nn
from torchvision import transforms
import scipy.stats as stats
import glob
import cv2
from skimage.metrics import structural_similarity as ssim
from colorama import Fore, Style



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
    abilities_dict = {'calloformmarius': 0, 'fromthedeep': 1, 'ormsmariuscharge': 2, 'powerofneptune': 3, 'tidalwave': 4, 'tridentstrike': 5, 'airatomizer': 6, 'atomfield': 7, 'makingmatter': 8, 'massiveslam': 9, 'bloodpush': 10, 'causticpool': 11, 'dexstarsrage': 12, 'pillarofblood': 13, 'siphonpower': 14, 'airtoprope': 15, 'bodypress': 16, 'doublepunch': 17, 'dropkick': 18, 'juggernautrush': 19, 'advancedparry': 20, 'airdownwardbatarang': 21, 'airstraightbatarang': 22, 'batstrike': 23, 'boomerangbat': 24, 'capeparry': 25, 'dualbatarangs': 26, 'evadingbat': 27, 'taserrangs': 28, 'airlightningcage': 29, 'deitysbolt': 30, 'powerofaton': 31, 'rollingthunder': 32, 'sethstrike': 33, 'soulofshazam': 34, 'aircanarycry': 35, 'bodybounce': 36, 'clutchcrush': 37, 'risingkick': 38, 'screech': 39, 'sonicburst': 40, 'blackpearl': 41, 'darksuit': 42, 'riptide': 43, 'sweepingmantarays': 44, 'aculeusstrike': 45, 'airdivinitymandiblestrike': 46, 'aliencloak': 47, 'mandibleflurry': 48, 'myrules': 49, 'destructionorb': 50, 'enchancedlivingmetal': 51, 'ioncannon': 52, 'tendrilharpoon': 53, 'tendrilslide': 54, 'aircoldblast': 55, 'airicebridge': 56, 'airiceout': 57, 'coldstream': 58, 'iceburst': 59, 'icerain': 60, 'shoulderofcold': 61, 'upwardcoldblast': 62, '9lives': 63, 'aircatslash': 64, 'catcall': 65, 'catclaws': 66, 'chaoticcat': 67, 'lowwhip': 68, 'aircheetahclutch': 69, 'airjunglejump': 70, 'creepingpredator': 71, 'savageslam': 72, 'spottedtorpedo': 73, 'bodyshield': 74, 'directedarmblaster': 75, 'airboomtubeaway': 76, 'boomrubeaway': 77, 'boomtube': 78, 'hateslam': 79, 'maximumomega': 80, 'reveresomegabeams': 81, 'airsniper': 82, 'heavyartillery': 83, 'lowwristcannon': 84, 'rocketjump': 85, 'scopedshot': 86, 'upshot': 87, 'absorptionspell': 88, 'airamonrablast': 89, 'airdash': 90, 'darkankh': 91, 'instantjudgement': 92, 'powerless': 93, 'seekingdisplacerorb': 94, 'anotherdimension': 95, 'banishingblast': 96, 'barrierspell': 97, 'demonsdissolve': 98, 'divinityspell': 99, 'eclipsosgrasp': 100, 'hypnoticspell': 101, 'airgroundspark': 102, 'atombomb': 103, 'flamephase': 104, 'meltingpoint': 105, 'vaporize': 106, 'airspinout': 107, 'fistsfrenzy': 108, 'lightningcharge': 109, 'sonicbolt': 110, 'sonicparry': 111, 'speednado': 112, 'doubledribble': 113, 'furiousflurry': 114, 'gorillagrab': 115, 'primitivepound': 116, 'bolaarrow': 117, 'distractingshot': 118, 'evasiveshock': 119, 'skybolt': 120, 'smokearrow': 121, 'airoasrocket': 122, 'lanternsbarrier': 123, 'minigun': 124, 'oasrocket': 125, 'quickcharge': 126, 'rocketpower': 127, 'turbinesmash': 128, 'allpurposefrosting': 129, 'cherrybomb': 130, 'confetticannon': 131, 'helovesme': 132, 'ivysblessing': 133, 'jokerinfection': 134, 'mollywhop': 135, 'ticktock': 136, 'airdevilsrevolver': 137, 'azzaelsguard': 138, 'brimstonegrenade': 139, 'gravedigger': 140, 'hellsfury': 141, 'crowbarcrush': 142, 'crowbarfling': 143, 'gasser': 144, 'knifeparry': 145, 'sideorderofpie': 146, 'surprise': 147, 'crawlingvines': 148, 'deadlythorns': 149, 'flowingearth': 150, 'petalsoflife': 151, 'thistlecoat': 152, 'airsparkport': 153, 'electriccurrent': 154, 'electricstorm': 155, 'powerbolt': 156, 'statictraps': 157, 'akimboblaze': 158, 'gutted': 159, 'hiddenexplosive': 160, 'instantdeath': 161, 'shrapnelblast': 162, 'timebomb': 163, 'airdeadlydirdarang': 164, 'deadlybirdarang': 165, 'elusiveswoop': 166, 'lineinthesand': 167, 'lowsmartbirdarang': 168, 'staffofgrayson': 169, 'floorflame': 170, 'panicportaway': 171, 'plague': 172, 'sacrifice': 173, 'soaringmurder': 174, 'terrorcharge': 175, 'floatingprincess': 176, 'novaburst': 177, 'spincycle': 178, 'starblastbarrage': 179, 'starslam': 180, 'xhalstrength': 181, 'airclonekick': 182, 'barrieroffrost': 183, 'groundfreeze': 184, 'hammerslam': 185, 'iceport': 186, 'klonecharge': 187, 'airheatvision': 188, 'earthshatter': 189, 'kryptoniangrinder': 190, 'speedingbullet': 191, 'suncharge': 192, 'airheatzap': 193, 'empoweredheatzap': 194, 'groundtremor': 195, 'kryptoncharge': 196, 'meteordrop': 197, 'truckpunchpull': 198, 'bonsai': 199, 'sinkingslough': 200, 'swampjuice': 201, 'pizzaparty': 202, 'raisingshell': 203, 'shellslide': 204, 'aegisofzeus': 205, 'airamazonianslam': 206, 'amaltheasprotection': 207, 'artemisstrength': 208, 'athenaspower': 209, 'demetersspirit': 210, 'hermesblessing': 211, 'hestiasgift': 212, 'lassospin': 213, 'swordofathena': 214, 'ability': 215, 'allpowerful': 216, 'almightypower': 217, 'augment': 218, 'boosted': 219, 'buddysystem': 220, 'deadlytransition': 221, 'eternallife': 222, 'feelthepain': 223, 'feelthepower': 224, 'gettingstronger': 225, 'godlike': 226, 'holdingback': 227, 'juggernaut': 228, 'liveforever': 229, 'longlife': 230, 'notdeadyet': 231, 'nothankyou': 232, 'notsostrong': 233, 'pumpedup': 234, 'steamroller': 235, 'strongerfaster': 236, 'tank': 237, 'truechampion': 238, 'whatiwant': 239}

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


def tensorize_db_record_abilities(db_input_list):

    attr_diff_norm_range = 1500

    fighter_1_name = torch.tensor([fighter_indices(fighter_name=db_input_list[0])])
    fighter_2_name = torch.tensor([fighter_indices(fighter_name=db_input_list[1])])

    fighter_1_level = torch.tensor([db_input_list[2]/30])
    fighter_2_level = torch.tensor([db_input_list[3]/30])

    fighter_1_attributes_not_norm = [int(n.replace('[', '').replace(']', '')) for n in db_input_list[4].split(',')]
    fighter_2_attributes_not_norm = [int(n.replace('[', '').replace(']', '')) for n in db_input_list[5].split(',')]

    attributes_difference = torch.tensor(normalize_list([a-b for a, b in zip(fighter_1_attributes_not_norm, fighter_2_attributes_not_norm)], min_max_range=(-1*attr_diff_norm_range, attr_diff_norm_range)))

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

    output = torch.cat([fighter_1_name, fighter_2_name, fighter_1_ability_1, fighter_1_ability_2, fighter_1_augment, fighter_2_ability_1, fighter_2_ability_2, fighter_2_augment, attributes_difference, fighter_1_ai, fighter_2_ai, advantage])

    return output.flatten()


def tensorize_db_record_abilities_inverted(db_input_list):
    attr_diff_norm_range = 1500

    fighter_1_name = torch.tensor([fighter_indices(fighter_name=db_input_list[1])])
    fighter_2_name = torch.tensor([fighter_indices(fighter_name=db_input_list[0])])

    fighter_1_level = torch.tensor([db_input_list[3]/30])
    fighter_2_level = torch.tensor([db_input_list[2]/30])

    fighter_1_attributes_not_norm = [int(n.replace('[', '').replace(']', '')) for n in db_input_list[5].split(',')]
    fighter_2_attributes_not_norm = [int(n.replace('[', '').replace(']', '')) for n in db_input_list[4].split(',')]

    attributes_difference = torch.tensor(
        normalize_list([a - b for a, b in zip(fighter_1_attributes_not_norm, fighter_2_attributes_not_norm)],
                       min_max_range=(-1 * attr_diff_norm_range, attr_diff_norm_range)))

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

    output = torch.cat([fighter_1_name, fighter_2_name, fighter_1_ability_1, fighter_1_ability_2, fighter_1_augment, fighter_2_ability_1, fighter_2_ability_2, fighter_2_augment, attributes_difference, fighter_1_ai, fighter_2_ai, advantage])
    return output.flatten()

def advantage_to_probabilities(prediction, std_dev):
    if prediction >= 0:
        probability_win = 1 - stats.norm.cdf(0, loc=prediction, scale=std_dev)
    else:
        # Calculate the cumulative distribution function
        probability_lose = stats.norm.cdf(0, loc=prediction, scale=std_dev)
        probability_win = 1 - probability_lose
    return probability_win

def score_encounter_predictions(p_raw):
    #p = normalize_list(advantage_list, min_max_range=(-1, 1))
    # check if the length of the list is 3

    p = [a.item() if a > 0 else 0 for a in p_raw]

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


        # Separate outputs with their respective activation functions
        x_winprob = self.activation(self.fc_winprob(x))
        x_advantage = self.activation(self.fc_advantage(x))
        out_winprob = self.sigmoid(self.fc_out1(x_winprob))
        out_advantage = self.tanh(self.fc_out2(x_advantage))

        return out_winprob, out_advantage

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


class ModelAdvantage_v2_5(nn.Module):
    def __init__(self):
        super(ModelAdvantage_v2_5, self).__init__()

        self.num_abilties = 240
        # Embedding layers
        self.fighter_name = nn.Embedding(38, 100)
        self.ability_embedding = nn.Embedding(self.num_abilties+1, 100)

        # Fully connected layers
        self.fc1 = nn.Linear(816, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc_winprob = nn.Linear(256, 128)
        self.fc_advantage = nn.Linear(256, 128)
        self.fc_out1 = nn.Linear(128, 1)  # Outputs a value between 0 and 1
        self.fc_out2 = nn.Linear(128, 1)  # Outputs a value between -1 and 1

        self.activation = nn.Mish()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.1)

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
        x = self.tanh(self.fc1(x))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.dropout(x)


        # Separate outputs with their respective activation functions
        x_winprob = self.activation(self.fc_winprob(x))
        x_advantage = self.activation(self.fc_advantage(x))
        out_winprob = self.sigmoid(self.fc_out1(x_winprob))
        out_advantage = self.tanh(self.fc_out2(x_advantage))

        return out_winprob, out_advantage