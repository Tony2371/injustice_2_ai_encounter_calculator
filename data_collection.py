import keyboard
import glob
import pyautogui
import numpy as np
import cv2
import time
import sqlite3
import hashlib
from classes_and_functions import Fighter, fighters_hash_pixels, template_matching, ModelHpTrack, ModelFighterRecognition, ModelDigitRecognition, flattened_masked_image, similarity, fighter_indices
import torch

#GLOBAL VARIABLES
folder_name = 'hp_extra_dataset'

conn = sqlite3.connect('injustice_2.db')
cursor = conn.cursor()

recording = False
match_image_dict = {}
player_selected_fighters = [None, None, None]
enemy_selected_fighters = [None, None, None]

hp_mask = cv2.imread('images/left_hp_mask.png', 0)
nn_hp_track = ModelHpTrack()
#nn_hp_track.load_state_dict(torch.load('nn_weights/weights_hp_track.pth'))
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
                           ai_primary='[0,18,20,18,4,0]',
                           ai_secondary='[0,0,30,30,0,0]',
                           attributes='[3169,2364,2027,2098]')

player_fighter_2 = Fighter(name='black_canary',
                           level=26,
                           ai_primary='[0,5,30,25,0,0]',
                           ai_secondary='[10,10,20,20,0,0]',
                           attributes='[1940,2635,1947,1662]')

player_fighter_3 = Fighter(name='bane',
                           level=16,
                           ai_primary='[8,2,20,30,0,0]',
                           ai_secondary='[0,8,30,22,0,0]',
                           attributes='[1953,1534,1640,1293]')

player_fighter_4 = Fighter(name='wonder_woman',
                           level=13,
                           ai_primary='[0,0,30,30,0,0]',
                           ai_secondary='[13,17,25,5,0,0]',
                           attributes='[1620,1535,1463,1118]')

player_fighter_5 = Fighter(name='enchantress',
                           level=10,
                           ai_primary='[0,13,30,17,0,0]',
                           ai_secondary='[0,0,28,0,25,7]',
                           attributes='[1395,1548,1294,1088]')

player_fighters = [player_fighter_1, player_fighter_2, player_fighter_3, player_fighter_4, player_fighter_5]



# MAIN LOOP
while True:
    keyboard.on_press_key('p',
                          lambda e: exec('global recording; recording = True'))
    keyboard.on_press_key('l',
                          lambda e: exec('global recording; recording = False'))


    if recording:
        collected_images = [image for image in glob.glob(folder_name+'/*.png')]

        screenshot = pyautogui.screenshot()
        screen_image = np.array(screenshot)

        screen_image = cv2.cvtColor(screen_image, cv2.COLOR_BGR2RGB)
        #screen_image = cv2.imread('images/ai_selection_3.png')
        zone_hp_1 = screen_image[50:90, 885:919]
        zone_hp_2 = screen_image[50:90, 1000:1035]

        zone_end_match_1 = screen_image[770:805, 572:632]
        zone_end_match_2 = screen_image[770:805, 1286:1351]

        # OPPONENT AI STATS PARSING ZONE
        if None in enemy_selected_fighters:
            if similarity(template_defending_team, screen_image[597:626, 1713:1901]) > 0.85:
                zone_opponent_1 = screen_image[291:291 + 80, 1425:1425 + 80]
                zone_opponent_2 = screen_image[291:291 + 80, 1593:1593 + 80]
                zone_opponent_3 = screen_image[291:291 + 80, 1758:1758 + 80]

                enemy_recognition_1 = nn_fighter_recognition(torch.tensor(zone_opponent_1 / 255))
                enemy_recognition_2 = nn_fighter_recognition(torch.tensor(zone_opponent_2 / 255))
                enemy_recognition_3 = nn_fighter_recognition(torch.tensor(zone_opponent_3 / 255))

                enemy_selected_fighters = [Fighter(name=fighter_indices(fighter_index=torch.argmax(enemy_recognition_1).item()), level=1),
                                            Fighter(name=fighter_indices(fighter_index=torch.argmax(enemy_recognition_2).item()), level=1),
                                            Fighter(name=fighter_indices(fighter_index=torch.argmax(enemy_recognition_3).item()), level=1)]


        zone_ai_parsing_icon = screen_image[140:170, 1870:1905]
        mask_zone_ai_parsing = cv2.inRange(zone_ai_parsing_icon, lowerb=(174, 172, 158), upperb=(174, 172, 158))

        if cv2.countNonZero(mask_zone_ai_parsing) == 94:

            fighter_name_zone = screen_image[146:172, 1620:1820]
            hash_current = hashlib.sha256(fighter_name_zone.tobytes()).hexdigest()
            for character, hash_value in fighters_hash_pixels.items():
                if hash_value == hash_current and None not in enemy_selected_fighters:
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

            #print([[a.name, a.selected_ai] for a in player_fighters])
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
            if torch.max(player_recognition_3).item() > 3:
                selection_3 = fighter_indices(fighter_index=torch.argmax(player_recognition_3).item())
                if selection_3 != 'not_selected' and len([f for f in player_fighters if f.name == selection_3]) > 0:
                    player_selected_fighters[2] = [f for f in player_fighters if f.name == selection_3][0]

            if None not in player_selected_fighters:
                print([(f.name, f.selected_ai) for f in player_selected_fighters])



        # MATCH END CONDITION
        mask_1 = cv2.inRange(zone_hp_1, lowerb=(0, 240, 248), upperb=(255, 242, 250))
        mask_2 = cv2.inRange(zone_hp_2, lowerb=(0, 240, 248), upperb=(255, 242, 250))

        # Fighter_1 looses
        if cv2.countNonZero(mask_1) >= 50:
            match_image_dict[int(time.time())] = screen_image
            zone_hp_right = screen_image[9:81, 999:1845]
            zone_hp_right = cv2.flip(zone_hp_right, 1)
            masked_hp_right = cv2.bitwise_and(zone_hp_right, zone_hp_right, mask=hp_mask)
            cv2.imwrite(f'debug_right.png', masked_hp_right)
            image_right = torch.tensor(
                flattened_masked_image(image=masked_hp_right, mask_path='images/left_hp_mask.png')).unsqueeze(0)
            hp_right = nn_hp_track(image_right).item()

            cv2.imwrite(folder_name + f'/{str(time.time()).replace(".", "_")}.png', screen_image)

            print(f'Recorded {int(time.time())}')
            print(f'Fighter_1 lost! Fighter_2 health: {hp_right}')
            time.sleep(0.5)

        # Fighter_2 looses
        if cv2.countNonZero(mask_2) >= 50:
            match_image_dict[int(time.time())] = screen_image
            zone_hp_left = screen_image[9:81, 71:917]
            masked_hp_left = cv2.bitwise_and(zone_hp_left, zone_hp_left, mask=hp_mask)
            cv2.imwrite(f'debug_left.png', masked_hp_left)
            image_left = torch.tensor(
                flattened_masked_image(image=masked_hp_left, mask_path='images/left_hp_mask.png')).unsqueeze(0)
            hp_left = nn_hp_track(image_left).item()

            cv2.imwrite(folder_name+f'/{str(time.time()).replace(".","_")}.png', screen_image)

            print(f'Recorded {int(time.time())}')
            print(f'Fighter_2 lost! Fighter_1 health: {hp_left}')
            time.sleep(0.5)

        # ENCOUNTER END CONDITION
        mask_end_1 = cv2.inRange(zone_end_match_1, lowerb=(82, 64, 49), upperb=(84, 66, 51))
        mask_end_2 = cv2.inRange(zone_end_match_2, lowerb=(82, 64, 49), upperb=(84, 66, 51))
        if cv2.countNonZero(mask_end_1) >= 0.95*zone_end_match_1.shape[0]*zone_end_match_1.shape[1] and cv2.countNonZero(mask_end_2) >= 0.95*zone_end_match_2.shape[0]*zone_end_match_2.shape[1]:
            match_image_dict[int(time.time())] = screen_image
            #cv2.imwrite(folder_name+f'/{str(time.time()).replace(".", "_")}.png', screen_image)
            for fighter_1, fighter_2 in zip(player_selected_fighters, enemy_selected_fighters):
                choose_ai = lambda f: f.ai_primary if f.selected_ai == 'primary' else f.ai_secondary
                data = (fighter_1.name, fighter_2.name, fighter_1.level, fighter_2.level, fighter_1.attributes, fighter_2.attributes, choose_ai(fighter_1), fighter_2.ai_primary)
                cursor.execute('INSERT INTO ai_battle_log_full (fighter_1_name, fighter_2_name, fighter_1_level, fighter_2_level, fighter_1_attributes, fighter_2_attributes,fighter_1_ai, fighter_2_ai) VALUES (?, ?, ?, ?, ?, ?, ?, ?)', data)
                print('Database record added')

            conn.commit()

            collected_images = [image for image in glob.glob(folder_name+'/*.png')]
            if len(collected_images) == 3:
                hp_bar_1 = cv2.imread(collected_images[0])[1:85, 0:1920]
                hp_bar_2 = cv2.imread(collected_images[1])[1:85, 0:1920]
                end_screen = cv2.imread(collected_images[2])[391:840, 0:1920]
                vertically_stacked = cv2.vconcat([hp_bar_1, hp_bar_2, end_screen])

            if len(collected_images) == 4:
                hp_bar_1 = cv2.imread(collected_images[0])[1:85, 0:1920]
                hp_bar_2 = cv2.imread(collected_images[1])[1:85, 0:1920]
                hp_bar_3 = cv2.imread(collected_images[2])[1:85, 0:1920]
                end_screen = cv2.imread(collected_images[3])[391:840, 0:1920]
                vertically_stacked = cv2.vconcat([hp_bar_1, hp_bar_2, hp_bar_3, end_screen])

            #cv2.imwrite(folder_name + '/stacked.png', vertically_stacked)

            #PARSING ENDING
            'RECORDED SUCCESSFULLY'
            recording=False
            time.sleep(5)

    if not recording:
        print('Stopped')
        match_image_dict = {}
        player_selected_fighters = [None, None, None]
        enemy_selected_fighters = [None, None, None]
        time.sleep(1)

