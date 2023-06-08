import keyboard
import glob
import pyautogui
import numpy as np
import cv2
import time
import sqlite3
import hashlib
from classes_and_functions import fighters_hash_pixels, template_matching, ModelHpTrack, flattened_masked_image
import torch

#GLOBAL VARIABLES
folder_name = 'encounter_temp'

conn = sqlite3.connect('injustice_2.db')
cursor = conn.cursor()

recording = False
ai_stats_dict = {}
match_image_dict = {}

hp_mask = cv2.imread('images/left_hp_mask.png', 0)
nn_hp_track = ModelHpTrack()
nn_hp_track.load_state_dict(torch.load('nn_weights/weights_hp_track.pth'))
nn_hp_track.eval()

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

        #AI STATS PARSING ZONE
        zone_ai_parsing_icon = screen_image[140:170, 1870:1905]
        mask_zone_ai_parsing = cv2.inRange(zone_ai_parsing_icon, lowerb=(174, 172, 158), upperb=(174, 172, 158))
        if cv2.countNonZero(mask_zone_ai_parsing) == 94:
            fighter_name_zone = screen_image[146:172, 1620:1820]
            hash_current = hashlib.sha256(fighter_name_zone.tobytes()).hexdigest()
            for character, hash_value in fighters_hash_pixels.items():
                if hash_value == hash_current:
                    if character not in ai_stats_dict.keys():

                        zone_grappling = screen_image[201:218, 1410:1432]
                        zone_rushdown = screen_image[245:262, 1410:1432]
                        zone_combos = screen_image[291:308, 1410:1432]
                        zone_counters = screen_image[336:353, 1410:1432]
                        zone_zoning = screen_image[381:398, 1410:1432]
                        zone_runaway = screen_image[426:443, 1410:1432]
                        grappling = template_matching(zone_grappling, 'at_stats_level_icons')
                        rushdown = template_matching(zone_rushdown, 'at_stats_level_icons')
                        combos = template_matching(zone_combos, 'at_stats_level_icons')
                        counters = template_matching(zone_counters, 'at_stats_level_icons')
                        zoning = template_matching(zone_zoning, 'at_stats_level_icons')
                        runaway = template_matching(zone_runaway, 'at_stats_level_icons')

                        ai_stats_dict[character] = str([grappling, rushdown, combos, counters, zoning, runaway]).replace(' ', '').replace("'", "")
                        print(ai_stats_dict)


        # MATCH END CONDITION
        mask_1 = cv2.inRange(zone_hp_1, lowerb=(0, 240, 248), upperb=(255, 242, 250))
        mask_2 = cv2.inRange(zone_hp_2, lowerb=(0, 240, 248), upperb=(255, 242, 250))

        # Fighter_1 looses
        if cv2.countNonZero(mask_1) >= 50:
            match_image_dict[int(time.time())] = screen_image
            zone_hp_left = screen_image[9:81, 71:917]
            masked_hp_left = cv2.bitwise_and(zone_hp_left, zone_hp_left, mask=hp_mask)
            image_left = torch.tensor(
                flattened_masked_image(image=masked_hp_left, mask_path='images/left_hp_mask.png')).unsqueeze(0)
            hp_left = nn_hp_track(image_left).item()

            cv2.imwrite(folder_name + f'/{str(time.time()).replace(".", "_")}.png', screen_image)

            print(f'Recorded {int(time.time())}')
            print(f'Fighter_1 lost! Fighter_1 health: {hp_left}')
            time.sleep(0.5)

        # Fighter_2 looses
        if cv2.countNonZero(mask_2) >= 50:
            match_image_dict[int(time.time())] = screen_image
            zone_hp_right = screen_image[9:81, 999:1845]
            zone_hp_right = cv2.flip(zone_hp_right, 1)
            masked_hp_right = cv2.bitwise_and(zone_hp_right, zone_hp_right, mask=hp_mask)
            image_right = torch.tensor(
                flattened_masked_image(image=masked_hp_right, mask_path='images/left_hp_mask.png')).unsqueeze(0)
            hp_right = nn_hp_track(image_right).item()

            cv2.imwrite(folder_name+f'/{str(time.time()).replace(".","_")}.png', screen_image)

            print(f'Recorded {int(time.time())}')
            print(f'Fighter_2 lost! Fighter_1 health: {hp_right}')
            time.sleep(0.5)

        # ENCOUNTER END CONDITION
        mask_end_1 = cv2.inRange(zone_end_match_1, lowerb=(82, 64, 49), upperb=(84, 66, 51))
        mask_end_2 = cv2.inRange(zone_end_match_2, lowerb=(82, 64, 49), upperb=(84, 66, 51))
        if cv2.countNonZero(mask_end_1) >= 0.95*zone_end_match_1.shape[0]*zone_end_match_1.shape[1] and cv2.countNonZero(mask_end_2) >= 0.95*zone_end_match_2.shape[0]*zone_end_match_2.shape[1]:
            match_image_dict[int(time.time())] = screen_image
            #cv2.imwrite(folder_name+f'/{str(time.time()).replace(".", "_")}.png', screen_image)
            for fighter_2 in ai_stats_dict.keys():
                cursor.execute('INSERT INTO ai_battle_log_advanced (fighter_2_name, fighter_2_ai) VALUES (?, ?)', (fighter_2, ai_stats_dict[fighter_2]))
                'Database record saved!'
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

            cv2.imwrite(folder_name + '/stacked.png', vertically_stacked)

            #PARSING ENDING
            time.sleep(5)

    if not recording:
        print('Stopped')
        ai_stats_dict = {}
        match_image_dict = {}
        time.sleep(1)

