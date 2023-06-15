import keyboard
import glob
import pyautogui
import numpy as np
import cv2
import time
import sqlite3
from classes_and_functions import Fighter, template_matching, ModelHpTrack, ModelFighterRecognition, ModelDigitRecognition, flattened_masked_image, similarity, fighter_indices
import torch

#GLOBAL VARIABLES
folder_name = 'for_hp_dataset'
seconds_per_tick = 1.0 / 30

conn = sqlite3.connect('injustice_2.db')
cursor = conn.cursor()

recording = False
printed = False
match_image_dict = {}
round_buffer = []
round_counter = 1
round_advantages = []
round_started = False
round_ended = False
player_selected_fighters = [None, None, None]
enemy_selected_fighters = [None, None, None]


nn_hp_track = ModelHpTrack()
nn_hp_track.load_state_dict(torch.load('nn_weights/weights_hp_track.pth'))
nn_hp_track.eval()

nn_fighter_recognition = ModelFighterRecognition()
nn_fighter_recognition.load_state_dict(torch.load('nn_weights/weights_fighter_recognition.pth'))
nn_fighter_recognition.eval()

nn_digit_recognition = ModelDigitRecognition()
nn_digit_recognition.load_state_dict(torch.load('nn_weights/weights_digit_recognition.pth'))
nn_digit_recognition.eval()

template_player_hero_selection = cv2.imread('images/template_player_hero_selection.png')
template_ai_primary = cv2.imread('images/template_primary.png')
template_ai_secondary = cv2.imread('images/template_secondary.png')
template_player_fighter_selection = cv2.imread('images/template_player_fighter_selection.png')
template_defending_team = cv2.imread('images/defending_team.png')
template_empty_digit = cv2.imread('images/template_empty_digit.png')


player_fighter_1 = Fighter(name='catwoman',
                           level=30,
                           ai_primary='[10,5,30,10,0,5]',
                           ai_secondary='[5,0,17,23,5,10]',
                           attributes='[3331,2296,1937,2102]')

player_fighter_2 = Fighter(name='black_canary',
                           level=28,
                           ai_primary='[0,5,30,25,0,0]',
                           ai_secondary='[5,15,30,10,0,0]',
                           attributes='[2224,2074,2158,1798]')

player_fighter_3 = Fighter(name='bane',
                           level=19,
                           ai_primary='[8,2,20,30,0,0]',
                           ai_secondary='[0,20,30,10,0,0]',
                           attributes='[2017,1551,2176,1387]')

player_fighter_4 = Fighter(name='wonder_woman',
                           level=17,
                           ai_primary='[5,15,15,25,0,0]',
                           ai_secondary='[0,15,25,20,0,0]',
                           attributes='[1854,1674,1532,1151]')

player_fighter_5 = Fighter(name='enchantress',
                           level=17,
                           ai_primary='[0,0,25,0,17,18]',
                           ai_secondary='[0,0,25,20,10,5]',
                           attributes='[1774,2103,1471,1165]')

player_fighter_6 = Fighter(name='supergirl',
                           level=13,
                           ai_primary='[5,10,25,20,0,0]',
                           ai_secondary='[0,15,15,30,0,0]',
                           attributes='[1547,1585,1656,1122]')

player_fighters = [player_fighter_1, player_fighter_2, player_fighter_3, player_fighter_4, player_fighter_5, player_fighter_6]



# MAIN LOOP
while True:
    start_time = time.time()
    keyboard.on_press_key('p',
                          lambda e: exec('global recording; recording = True'))
    keyboard.on_press_key('l',
                          lambda e: exec('global recording; recording = False'))


    if recording:
        collected_images = [image for image in glob.glob(folder_name+'/*.png')]

        screenshot = pyautogui.screenshot()
        screen_image = np.array(screenshot)

        screen_image = cv2.cvtColor(screen_image, cv2.COLOR_BGR2RGB)

        zone_hp_1 = screen_image[50:90, 885:919]
        zone_hp_2 = screen_image[50:90, 1000:1035]

        zone_end_match_1 = screen_image[770:805, 572:632]
        zone_end_match_2 = screen_image[770:805, 1286:1351]

        # OPPONENT AI STATS PARSING ZONE
        if None in enemy_selected_fighters:
            if similarity(template_defending_team, screen_image[597:626, 1713:1901]) > 0.85:
                time.sleep(0.3)
                zone_opponent_1 = screen_image[291:291 + 80, 1425:1425 + 80]
                zone_opponent_2 = screen_image[291:291 + 80, 1593:1593 + 80]
                zone_opponent_3 = screen_image[291:291 + 80, 1758:1758 + 80]

                opponent_recognition_1 = nn_fighter_recognition(torch.tensor(zone_opponent_1 / 255))
                opponent_recognition_2 = nn_fighter_recognition(torch.tensor(zone_opponent_2 / 255))
                opponent_recognition_3 = nn_fighter_recognition(torch.tensor(zone_opponent_3 / 255))

                enemy_selected_fighters = [Fighter(name=fighter_indices(fighter_index=torch.argmax(opponent_recognition_1).item()), level=1),
                                            Fighter(name=fighter_indices(fighter_index=torch.argmax(opponent_recognition_2).item()), level=1),
                                            Fighter(name=fighter_indices(fighter_index=torch.argmax(opponent_recognition_3).item()), level=1)]

                zone_opponent_1_level_1 = screen_image[513:535, 1407:1421]
                zone_opponent_1_level_2 = screen_image[513:535, 1420:1434]
                opponent_1_level_recognition_1 = nn_digit_recognition(torch.tensor(zone_opponent_1_level_1/255).unsqueeze(0))
                opponent_1_level_recognition_2 = nn_digit_recognition(torch.tensor(zone_opponent_1_level_2/255).unsqueeze(0))

                zone_opponent_2_level_1 = screen_image[513:535, 1574:1588]
                zone_opponent_2_level_2 = screen_image[513:535, 1587:1601]
                opponent_2_level_recognition_1 = nn_digit_recognition(
                    torch.tensor(zone_opponent_2_level_1 / 255).unsqueeze(0))
                opponent_2_level_recognition_2 = nn_digit_recognition(
                    torch.tensor(zone_opponent_2_level_2 / 255).unsqueeze(0))

                zone_opponent_3_level_1 = screen_image[513:535, 1739:1753]
                zone_opponent_3_level_2 = screen_image[513:535, 1752:1766]
                opponent_3_level_recognition_1 = nn_digit_recognition(
                    torch.tensor(zone_opponent_3_level_1 / 255).unsqueeze(0))
                opponent_3_level_recognition_2 = nn_digit_recognition(
                    torch.tensor(zone_opponent_3_level_2 / 255).unsqueeze(0))


                if torch.max(opponent_1_level_recognition_2).item() < 6:
                    enemy_selected_fighters[0].level = torch.argmax(opponent_1_level_recognition_1).item()
                else:
                    enemy_selected_fighters[0].level = int(str(torch.argmax(opponent_1_level_recognition_1).item())+str(torch.argmax(opponent_1_level_recognition_2).item()))

                if torch.max(opponent_2_level_recognition_2).item() < 6:
                    enemy_selected_fighters[1].level = torch.argmax(opponent_2_level_recognition_1).item()
                else:
                    enemy_selected_fighters[1].level = int(str(torch.argmax(opponent_2_level_recognition_1).item())+str(torch.argmax(opponent_2_level_recognition_2).item()))

                if torch.max(opponent_3_level_recognition_2).item() < 6:
                    enemy_selected_fighters[2].level = torch.argmax(opponent_3_level_recognition_1).item()
                else:
                    enemy_selected_fighters[2].level = int(str(torch.argmax(opponent_3_level_recognition_1).item())+str(torch.argmax(opponent_3_level_recognition_2).item()))



                #cv2.imwrite(f'digits_dataset/Z1{str(time.time()).split(".")[1]}.png', zone_opponent_1_level_1)
                #cv2.imwrite(f'digits_dataset/Z2{str(time.time()).split(".")[1]}.png', zone_opponent_1_level_2)
                #cv2.imwrite(f'digits_dataset/Z3{str(time.time()).split(".")[1]}.png', zone_opponent_2_level_1)
                #cv2.imwrite(f'digits_dataset/Z4{str(time.time()).split(".")[1]}.png', zone_opponent_2_level_2)
                #cv2.imwrite(f'digits_dataset/Z5{str(time.time()).split(".")[1]}.png', zone_opponent_3_level_1)
                #cv2.imwrite(f'digits_dataset/Z6{str(time.time()).split(".")[1]}.png', zone_opponent_3_level_2)



                print([f.level for f in enemy_selected_fighters])


        zone_ai_parsing_icon = screen_image[140:170, 1870:1905]
        mask_zone_ai_parsing = cv2.inRange(zone_ai_parsing_icon, lowerb=(174, 172, 158), upperb=(174, 172, 158))

        if cv2.countNonZero(mask_zone_ai_parsing) == 94 and None not in enemy_selected_fighters:
            fighter_name_zone = screen_image[146:172, 1620:1820]
            character, _ = template_matching(fighter_name_zone, 'template_opponent_fighter_names')
            for fighter in enemy_selected_fighters:
                if fighter.name == character:
                    # GRAPPLING STAT DETECTION
                    grappling_value_1 = "_"
                    grappling_value_2 = "_"
                    zone_grappling_1 = screen_image[198:198+22, 1408:1408+14]
                    zone_grappling_2 = screen_image[198:198+22, 1420:1420+14]
                    grappling_recognition_1 = nn_digit_recognition(torch.tensor(zone_grappling_1/255).unsqueeze(0))
                    grappling_recognition_2 = nn_digit_recognition(torch.tensor(zone_grappling_2/255).unsqueeze(0))
                    if torch.max(grappling_recognition_1).item() > 4.5:
                        grappling_value_1 = str(torch.argmax(grappling_recognition_1).item())
                    if torch.max(grappling_recognition_2).item() > 4.5:
                        grappling_value_2 = str(torch.argmax(grappling_recognition_2).item())
                    if similarity(template_empty_digit, zone_grappling_2) < 0.85:
                        grappling = grappling_value_1+grappling_value_2
                    else:
                        grappling = grappling_value_1


                    # RUSHDOWN STAT DETECTION
                    rushdown_value_1 = "_"
                    rushdown_value_2 = "_"
                    zone_rushdown_1 = screen_image[242:242 + 22, 1408:1408 + 14]
                    zone_rushdown_2 = screen_image[242:242 + 22, 1420:1420 + 14]
                    rushdown_recognition_1 = nn_digit_recognition(
                        torch.tensor(zone_rushdown_1 / 255).unsqueeze(0))
                    rushdown_recognition_2 = nn_digit_recognition(
                        torch.tensor(zone_rushdown_2 / 255).unsqueeze(0))
                    if torch.max(rushdown_recognition_1).item() > 4.5:
                        rushdown_value_1 = str(torch.argmax(rushdown_recognition_1).item())
                    if torch.max(rushdown_recognition_2).item() > 4.5:
                        rushdown_value_2 = str(torch.argmax(rushdown_recognition_2).item())
                    if similarity(template_empty_digit, zone_rushdown_2) < 0.85:
                        rushdown = rushdown_value_1 + rushdown_value_2
                    else:
                        rushdown = rushdown_value_1


                    # COMBOS ZONE DETECTION
                    combos_value_1 = "_"
                    combos_value_2 = "_"
                    zone_combos_1 = screen_image[288:288 + 22, 1408:1408 + 14]
                    zone_combos_2 = screen_image[288:288 + 22, 1420:1420 + 14]
                    combos_recognition_1 = nn_digit_recognition(
                        torch.tensor(zone_combos_1 / 255).unsqueeze(0))
                    combos_recognition_2 = nn_digit_recognition(
                        torch.tensor(zone_combos_2 / 255).unsqueeze(0))
                    if torch.max(combos_recognition_1).item() > 4.5:
                        combos_value_1 = str(torch.argmax(combos_recognition_1).item())
                    if torch.max(combos_recognition_2).item() > 4.5:
                        combos_value_2 = str(torch.argmax(combos_recognition_2).item())
                    if similarity(template_empty_digit, zone_combos_2) < 0.85:
                        combos = combos_value_1 + combos_value_2
                    else:
                        combos = combos_value_1


                    # COUNTERS ZONE DETECTION
                    counters_value_1 = "_"
                    counters_value_2 = "_"
                    zone_counters_1 = screen_image[333:333 + 22, 1408:1408 + 14]
                    zone_counters_2 = screen_image[333:333 + 22, 1420:1420 + 14]
                    counters_recognition_1 = nn_digit_recognition(
                        torch.tensor(zone_counters_1 / 255).unsqueeze(0))
                    counters_recognition_2 = nn_digit_recognition(
                        torch.tensor(zone_counters_2 / 255).unsqueeze(0))
                    if torch.max(counters_recognition_1).item() > 4.5:
                        counters_value_1 = str(torch.argmax(counters_recognition_1).item())
                    if torch.max(counters_recognition_2).item() > 4.5:
                        counters_value_2 = str(torch.argmax(counters_recognition_2).item())
                    if similarity(template_empty_digit, zone_counters_2) < 0.85:
                        counters = counters_value_1 + counters_value_2
                    else:
                        counters = counters_value_1


                    # ZONING ZONE DETECTION
                    zoning_value_1 = "_"
                    zoning_value_2 = "_"
                    zone_zoning_1 = screen_image[378:378 + 22, 1408:1408 + 14]
                    zone_zoning_2 = screen_image[378:378 + 22, 1420:1420 + 14]
                    zoning_recognition_1 = nn_digit_recognition(
                        torch.tensor(zone_zoning_1 / 255).unsqueeze(0))
                    zoning_recognition_2 = nn_digit_recognition(
                        torch.tensor(zone_zoning_2 / 255).unsqueeze(0))
                    if torch.max(zoning_recognition_1).item() > 4.5:
                        zoning_value_1 = str(torch.argmax(zoning_recognition_1).item())
                    if torch.max(zoning_recognition_2).item() > 4.5:
                        zoning_value_2 = str(torch.argmax(zoning_recognition_2).item())
                    if similarity(template_empty_digit, zone_zoning_2) < 0.85:
                        zoning = zoning_value_1 + zoning_value_2
                    else:
                        zoning = zoning_value_1


                    # RUNAWAY ZONE DETECTION
                    runaway_value_1 = "_"
                    runaway_value_2 = "_"
                    zone_runaway_1 = screen_image[423:423 + 22, 1408:1408 + 14]
                    zone_runaway_2 = screen_image[423:423 + 22, 1420:1420 + 14]
                    runaway_recognition_1 = nn_digit_recognition(
                        torch.tensor(zone_runaway_1 / 255).unsqueeze(0))
                    runaway_recognition_2 = nn_digit_recognition(
                        torch.tensor(zone_runaway_2 / 255).unsqueeze(0))
                    if torch.max(runaway_recognition_1).item() > 4.5:
                        runaway_value_1 = str(torch.argmax(runaway_recognition_1).item())
                    if torch.max(runaway_recognition_2).item() > 4.5:
                        runaway_value_2 = str(torch.argmax(runaway_recognition_2).item())
                    if similarity(template_empty_digit, zone_runaway_2) < 0.85:
                        runaway = runaway_value_1 + runaway_value_2
                    else:
                        runaway = runaway_value_1

                    fighter.ai_primary = str([grappling, rushdown, combos, counters, zoning, runaway]).replace(' ', '').replace("'", "")


                    # OPPONENT ATTRIBUTES PARSING ZONE
                    str_recognition_1 = nn_digit_recognition(
                        torch.tensor(cv2.resize(screen_image[81:107, 1630:1646], (14, 22))).unsqueeze(0))
                    str_recognition_2 = nn_digit_recognition(
                        torch.tensor(cv2.resize(screen_image[81:107, 1646:1662], (14, 22))).unsqueeze(0))
                    str_recognition_3 = nn_digit_recognition(
                        torch.tensor(cv2.resize(screen_image[81:107, 1660:1676], (14, 22))).unsqueeze(0))
                    str_recognition_4 = nn_digit_recognition(
                        torch.tensor(cv2.resize(screen_image[81:107, 1674:1690], (14, 22))).unsqueeze(0))

                    strength = str(torch.argmax(str_recognition_1).item())+\
                    str(torch.argmax(str_recognition_2).item())+\
                    str(torch.argmax(str_recognition_3).item())+\
                    str(torch.argmax(str_recognition_4).item())


                    abl_recognition_1 = nn_digit_recognition(
                        torch.tensor(cv2.resize(screen_image[81:107, 1701:1717], (14, 22))).unsqueeze(0))
                    abl_recognition_2 = nn_digit_recognition(
                        torch.tensor(cv2.resize(screen_image[81:107, 1716:1732], (14, 22))).unsqueeze(0))
                    abl_recognition_3 = nn_digit_recognition(
                        torch.tensor(cv2.resize(screen_image[81:107, 1730:1746], (14, 22))).unsqueeze(0))
                    abl_recognition_4 = nn_digit_recognition(
                        torch.tensor(cv2.resize(screen_image[81:107, 1744:1760], (14, 22))).unsqueeze(0))

                    ability = str(torch.argmax(abl_recognition_1).item()) + \
                               str(torch.argmax(abl_recognition_2).item()) + \
                               str(torch.argmax(abl_recognition_3).item()) + \
                               str(torch.argmax(abl_recognition_4).item())


                    def_recognition_1 = nn_digit_recognition(
                        torch.tensor(cv2.resize(screen_image[81:107, 1771:1771+16], (14, 22))).unsqueeze(0))
                    def_recognition_2 = nn_digit_recognition(
                        torch.tensor(cv2.resize(screen_image[81:107, 1786:1786+16], (14, 22))).unsqueeze(0))
                    def_recognition_3 = nn_digit_recognition(
                        torch.tensor(cv2.resize(screen_image[81:107, 1800:1816], (14, 22))).unsqueeze(0))
                    def_recognition_4 = nn_digit_recognition(
                        torch.tensor(cv2.resize(screen_image[81:107, 1814:1830], (14, 22))).unsqueeze(0))

                    defense = str(torch.argmax(def_recognition_1).item()) + \
                              str(torch.argmax(def_recognition_2).item()) + \
                              str(torch.argmax(def_recognition_3).item()) + \
                              str(torch.argmax(def_recognition_4).item())


                    hp_recognition_1 = nn_digit_recognition(
                        torch.tensor(cv2.resize(screen_image[81:107, 1841:1857], (14, 22))).unsqueeze(0))
                    hp_recognition_2 = nn_digit_recognition(
                        torch.tensor(cv2.resize(screen_image[81:107, 1856:1856 + 16], (14, 22))).unsqueeze(0))
                    hp_recognition_3 = nn_digit_recognition(
                        torch.tensor(cv2.resize(screen_image[81:107, 1870:1886], (14, 22))).unsqueeze(0))
                    hp_recognition_4 = nn_digit_recognition(
                        torch.tensor(cv2.resize(screen_image[81:107, 1884:1900], (14, 22))).unsqueeze(0))

                    health_points = str(torch.argmax(hp_recognition_1).item()) + \
                              str(torch.argmax(hp_recognition_2).item()) + \
                              str(torch.argmax(hp_recognition_3).item()) + \
                              str(torch.argmax(hp_recognition_4).item())


                    #cv2.imwrite('digits_dataset/Z_digit.png', cv2.resize(screen_image[81:107, 1701:1717], (14, 22)))
                    #cv2.imwrite('digits_dataset/Z1_digit.png', cv2.resize(screen_image[81:107, 1856:1856 + 16], (14, 22)))
                    #cv2.imwrite('digits_dataset/Z1_digit.png', cv2.resize(screen_image[81:107, 1730:1746], (14, 22)))
                    #cv2.imwrite('digits_dataset/Z1_digit.png', cv2.resize(screen_image[81:107, 1744:1760], (14, 22)))
                    '''
                    cv2.imwrite(f'digits_dataset/w_{int(time.time())}.png',
                                cv2.resize(screen_image[81:107, 1841:1857], (14, 22)))
                    cv2.imwrite(f'digits_dataset/x_{int(time.time())}.png',
                                cv2.resize(screen_image[81:107, 1856:1856 + 16], (14, 22)))
                    cv2.imwrite(f'digits_dataset/y_{int(time.time())}.png',
                                cv2.resize(screen_image[81:107, 1870:1886], (14, 22)))
                    cv2.imwrite(f'digits_dataset/z_{int(time.time())}.png',
                                cv2.resize(screen_image[81:107, 1884:1900], (14, 22)))
                    
                    '''

                    fighter.attributes = f'[{strength},{ability},{defense},{health_points}]'
                    print(fighter.attributes)


        # PLAYER AI STATS PARSING ZONE
        if similarity(screen_image[55:75, 110:370], template_player_hero_selection) >= 0.85:
            for fighter in player_fighters:
                if fighter.name == template_matching(screen_image[0:44, 0:300], 'player_fighter_names')[0]:
                    current_selected_fighter = fighter.name
            if similarity(screen_image[148:173, 78:250], template_ai_secondary) >= 0.85:
                for f in player_fighters:
                    if f.name == current_selected_fighter:
                        f.selected_ai = 'secondary'
            if similarity(screen_image[148:173, 78:250], template_ai_primary) >= 0.85:
                for f in player_fighters:
                    if f.name == current_selected_fighter:
                        f.selected_ai = 'primary'

            zone_player_1 = screen_image[311:311+80, 116:116+80]
            zone_player_2 = screen_image[311:311+80, 287:287+80]
            zone_player_3 = screen_image[311:311+80, 448:448+80]

            player_recognition_1 = nn_fighter_recognition(torch.tensor(zone_player_1 / 255))
            player_recognition_2 = nn_fighter_recognition(torch.tensor(zone_player_2 / 255))
            player_recognition_3 = nn_fighter_recognition(torch.tensor(zone_player_3 / 255))

            if torch.max(player_recognition_1).item() > 3:
                selection_1 = fighter_indices(fighter_index=torch.argmax(player_recognition_1).item())
                if selection_1 != 'not_selected' and len([f for f in player_fighters if f.name == selection_1]) > 0:
                    player_selected_fighters[0] = [f for f in player_fighters if f.name == selection_1][0]
            if torch.max(player_recognition_2).item() > 3:
                selection_2 = fighter_indices(fighter_index=torch.argmax(player_recognition_2).item())
                if selection_2 != 'not_selected' and len([f for f in player_fighters if f.name == selection_2]) > 0:
                    player_selected_fighters[1] = [f for f in player_fighters if f.name == selection_2][0]
            if cv2.countNonZero(cv2.inRange(screen_image[600:650, 1380:1430], lowerb=(0, 0, 0), upperb=(10, 10, 10))) >= 200:
                selection_3 = template_matching(screen_image[0:44, 0:300], 'player_fighter_names')[0]
                player_selected_fighters[2] = [f for f in player_fighters if f.name == selection_3][0]
                if None not in player_selected_fighters:
                    print([(f.name, f.selected_ai) for f in player_selected_fighters])

        # ROUND IN PROGRESS
        zone_round_left = screen_image[60:80, 927:937]
        zone_round_right = screen_image[60:80, 982:992]

        detection_zone_round_left = cv2.countNonZero(cv2.inRange(zone_round_left, lowerb=(104, 72, 67), upperb=(115, 78, 72)))
        detection_zone_round_right = cv2.countNonZero(cv2.inRange(zone_round_right, lowerb=(104, 72, 67), upperb=(115, 78, 72)))

        if detection_zone_round_left >= 185 and detection_zone_round_right >= 185:
            fighter_1_hp_zone = screen_image[9:79, 73:917]
            fighter_2_hp_zone = cv2.flip(screen_image[9:79, 1004:1848], 1)

            fighter_1_hp = nn_hp_track(
                torch.tensor(fighter_1_hp_zone / 255, dtype=torch.float).permute(2, 0, 1).unsqueeze(0)).item()
            fighter_2_hp = nn_hp_track(
                torch.tensor(fighter_2_hp_zone / 255, dtype=torch.float).permute(2, 0, 1).unsqueeze(0)).item()

        round_buffer.append(detection_zone_round_left)
        round_buffer.append(detection_zone_round_right)
        if len(round_buffer) >= 180:
            round_buffer.pop(0)
            round_buffer.pop(0)

        if sum(round_buffer)/len(round_buffer) >= 120:
            round_started = True
            round_ended = False
        else:
            if not round_ended and round_started == True:

                if fighter_1_hp > fighter_2_hp:
                    round_advantages.append(round(fighter_1_hp, 3))
                else:
                    round_advantages.append(round(fighter_2_hp, 3)*-1)

                print(f'Round {round_counter} ended')
                print('Fighter 1 hp: ', round(fighter_1_hp, 3))
                print('Fighter 2 hp: ', round(fighter_2_hp, 3))
                cv2.imwrite(folder_name + f'/neural_{str(time.time()).replace(".", "_")}_{round_counter}_l.png',
                            fighter_1_hp_zone)
                cv2.imwrite(folder_name + f'/neural_{str(time.time()).replace(".", "_")}_{round_counter}_r.png',
                            fighter_2_hp_zone)

                round_counter += 1
                round_ended = True
                round_started = False


        # ENCOUNTER END CONDITION
        mask_end_1 = cv2.inRange(zone_end_match_1, lowerb=(82, 64, 49), upperb=(84, 66, 51))
        mask_end_2 = cv2.inRange(zone_end_match_2, lowerb=(82, 64, 49), upperb=(84, 66, 51))
        if cv2.countNonZero(mask_end_1) >= 0.95*zone_end_match_1.shape[0]*zone_end_match_1.shape[1] and cv2.countNonZero(mask_end_2) >= 0.95*zone_end_match_2.shape[0]*zone_end_match_2.shape[1]:
            match_image_dict[int(time.time())] = screen_image
            # 3 for two rounds, becase we start with 2
            if round_counter == 3:
                for fighter_1, fighter_2, advantage in zip(player_selected_fighters[:2], enemy_selected_fighters[:2], round_advantages):
                    try:
                        choose_ai = lambda f: f.ai_primary if f.selected_ai == 'primary' else f.ai_secondary
                        data = (fighter_1.name, fighter_2.name, fighter_1.level, fighter_2.level, fighter_1.attributes, fighter_2.attributes, choose_ai(fighter_1), fighter_2.ai_primary, advantage)
                        cursor.execute('INSERT INTO ai_battle_log_full (fighter_1_name, fighter_2_name, fighter_1_level, fighter_2_level, fighter_1_attributes, fighter_2_attributes,fighter_1_ai, fighter_2_ai, result) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)', data)
                        print('Database record added')
                    except Exception as e:
                        print(e)
                        print(player_selected_fighters)
                        print(enemy_selected_fighters)
            # 4 for three rounds, becase we start with 2
            if round_counter == 4:
                for fighter_1, fighter_2, advantage in zip(player_selected_fighters, enemy_selected_fighters, round_advantages):
                    try:
                        choose_ai = lambda f: f.ai_primary if f.selected_ai == 'primary' else f.ai_secondary
                        data = (fighter_1.name, fighter_2.name, fighter_1.level, fighter_2.level, fighter_1.attributes, fighter_2.attributes, choose_ai(fighter_1), fighter_2.ai_primary, advantage)
                        cursor.execute('INSERT INTO ai_battle_log_full (fighter_1_name, fighter_2_name, fighter_1_level, fighter_2_level, fighter_1_attributes, fighter_2_attributes,fighter_1_ai, fighter_2_ai, result) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)', data)
                        print('Database record added')
                    except:
                        print(player_selected_fighters)
                        print(enemy_selected_fighters)


            conn.commit()

            #PARSING ENDING
            print('Encounter saved ')
            recording = False


    if not recording:
        match_image_dict = {}
        printed = False
        player_selected_fighters = [None, None, None]
        enemy_selected_fighters = [None, None, None]
        round_buffer = []
        round_counter = 1
        round_advantages = []



    end_time = time.time()
    elapsed_time = end_time - start_time
    if elapsed_time < seconds_per_tick:
        time.sleep(seconds_per_tick - elapsed_time)

