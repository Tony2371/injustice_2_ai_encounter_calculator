import cv2
import numpy as np
import os
import glob
import pandas as pd
import torch
from colorama import Fore, Style
import itertools
import sqlite3
import pyautogui
import time

from classes_and_functions import Row, Fighter, Net, Encounter_group


conn = sqlite3.connect('injustice_2.db')


def calculate_probs(image):
    img = cv2.imread(image)

    row_1 = Row(img[193:271, 821:1094], 1)
    row_2 = Row(img[290:370, 821:1094], 2)
    row_3 = Row(img[390:470, 821:1094], 3)
    row_4 = Row(img[490:570, 821:1094], 4)
    row_5 = Row(img[590:670, 821:1094], 5)
    row_6 = Row(img[690:770, 821:1094], 6)
    row_7 = Row(img[790:870, 821:1094], 7)
    row_8 = Row(img[890:970, 821:1094], 8)

    rows = [row_1, row_2, row_3, row_4, row_5, row_6, row_7, row_8]
    character_icons = glob.glob('character_icons/*.png')
    level_icons = glob.glob('level_icons/*.png')

    for row in rows:
        # FIND CHARACTER MATCHES
        for img_path in character_icons:
            icon = cv2.imread(img_path)
            character_name = os.path.basename(img_path)

            res = cv2.matchTemplate(row.image, icon, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            if max_val > 0.6:
                if max_loc[0] > 185:
                    row.characters[2] = character_name.replace('.png', '')
                elif max_loc[0] > 95:
                    row.characters[1] = character_name.replace('.png', '')
                else:
                    row.characters[0] = character_name.replace('.png', '')

            # FIND LEVEL MATCHES
        height, width, _ = row.image.shape
        size = width // 3
        for i, chunk in enumerate([row.image[:, 0:size], row.image[:, size:2 * size], row.image[:, 2 * size:3 * size]]):
            lvl_names = []
            lvl_probs = []
            for img_path in level_icons:

                icon = cv2.imread(img_path)
                level = os.path.basename(img_path)

                res = cv2.matchTemplate(chunk, icon, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

                lvl_names.append(level)
                lvl_probs.append(max_val)

                index = lvl_probs.index(max(lvl_probs))
                result = int(lvl_names[index].replace('.png', ''))

            row.levels[i] = result


    model = Net()
    model.load_state_dict(torch.load(('injustice_model.pth')))
    model.eval()

    df = pd.read_sql(sql='SELECT * FROM ai_battle_log', con=conn)
    _, fighter_indices = pd.factorize(df['fighter_2_name'])

    for row in rows:
        print(f'ROW {row.index}')
        encounter_groups = []
        fighters_1_list = [Fighter('catwoman', 30), Fighter('black_canary', 26), Fighter('enchantress', 9), Fighter('wonder_woman', 11), Fighter('bane', 14)]
        fighters_2_list = [Fighter(row.characters[0], row.levels[0]), Fighter(row.characters[1], row.levels[1]), Fighter(row.characters[2], row.levels[2])]

        player_fighter_combination = list(itertools.combinations(fighters_1_list, 3))

        encounters_all = []

        for combination_of_three in player_fighter_combination:
            encounters = list(itertools.product(combination_of_three, fighters_2_list))
            encounters_all.append(Encounter_group(group=[encounters[0], encounters[4], encounters[8]], model=model, fighter_indices=fighter_indices))
            encounters_all.append(Encounter_group(group=[encounters[1], encounters[5], encounters[6]], model=model, fighter_indices=fighter_indices))
            encounters_all.append(Encounter_group(group=[encounters[2], encounters[3], encounters[7]], model=model, fighter_indices=fighter_indices))


        row_max_win_chance = max([group.total_prob_win for group in encounters_all])
        winning_group_index = [group.total_prob_win for group in encounters_all].index(row_max_win_chance)
        winning_group = encounters_all[winning_group_index]
        print(f'Encounter_1: {winning_group.encounter_1[0].name} {winning_group.encounter_1[0].level} vs.{winning_group.encounter_1[1].name} {winning_group.encounter_1[1].level} | Win chance: {winning_group.encounter_1_win_chance*100} %')
        print(f'Encounter_2: {winning_group.encounter_2[0].name} {winning_group.encounter_2[0].level} vs.{winning_group.encounter_2[1].name} {winning_group.encounter_2[1].level} | Win chance: {winning_group.encounter_2_win_chance*100} %')
        print(f'Encounter_3: {winning_group.encounter_3[0].name} {winning_group.encounter_3[0].level} vs.{winning_group.encounter_3[1].name} {winning_group.encounter_3[1].level} | Win chance: {winning_group.encounter_3_win_chance*100} %')
        print(f'TOTAL WIN CHANCE: {winning_group.total_prob_win*100} %')

while True:
    reference = cv2.imread('images/encounter.png', cv2.COLOR_BGR2RGB)
    screenshot = pyautogui.screenshot()
    screen_image = np.array(screenshot)
    difference = cv2.subtract(screen_image, reference)
    difference = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    non_zero_pixels_left = cv2.countNonZero(difference)
    total_pixels = difference.shape[0] * difference.shape[1]
    similarity = (total_pixels - non_zero_pixels_left) / total_pixels * 100
    #print(similarity)

    calculate_probs('images/encounter.png')
    break


