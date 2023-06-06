import keyboard
import glob
import pyautogui
import numpy as np
import cv2
import time
import sqlite3
import hashlib
from classes_and_functions import fighters_hash_pixels, template_matching

#GLOBAL VARIABLES
folder_name = 'encounter_temp'

conn = sqlite3.connect('injustice_2.db')
cursor = conn.cursor()

recording = False
temp_enemy_character_track = []
temp_enemy_ai_stats_track = []

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
                    if character not in temp_enemy_character_track:
                        temp_enemy_character_track.append(character)

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

                        temp_enemy_ai_stats_track.append([grappling, rushdown, combos, counters, zoning, runaway])


        print(temp_enemy_character_track)
        print(temp_enemy_ai_stats_track)


        mask_1 = cv2.inRange(zone_hp_1, lowerb=(0, 240, 248), upperb=(255, 242, 250))
        mask_2 = cv2.inRange(zone_hp_2, lowerb=(0, 240, 248), upperb=(255, 242, 250))

        mask_end_1 = cv2.inRange(zone_end_match_1, lowerb=(82, 64, 49), upperb=(84, 66, 51))
        mask_end_2 = cv2.inRange(zone_end_match_2, lowerb=(82, 64, 49), upperb=(84, 66, 51))

        if cv2.countNonZero(mask_1) >= 50 or cv2.countNonZero(mask_2) >= 50:
            cv2.imwrite(folder_name+f'/{str(time.time()).replace(".","_")}.png', screen_image)
            print(f'Recorded {int(time.time())}')
            time.sleep(0.5)

        if cv2.countNonZero(mask_end_1) >= 0.95*zone_end_match_1.shape[0]*zone_end_match_1.shape[1] and cv2.countNonZero(mask_end_2) >= 0.95*zone_end_match_2.shape[0]*zone_end_match_2.shape[1]:
            cv2.imwrite(folder_name+f'/{str(time.time()).replace(".", "_")}.png', screen_image)

            for fighter_2, fighter_2_ai in zip(temp_enemy_character_track, temp_enemy_ai_stats_track):
                cursor.execute('INSERT INTO ai_battle_log_advanced (fighter_2_name, fighter_2_ai) VALUES (?, ?)', (fighter_2, str(fighter_2_ai).replace(' ', '')))
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

            recording = False
            conn.close()
            time.sleep(5)

    if not recording:
        print('Stopped')
        temp_enemy_character_track = []
        temp_enemy_ai_stats_track = []
        time.sleep(1)

