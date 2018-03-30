import json
import os
import cv2 as cv
from shutil import copyfile
import numpy as np

#scene_list=["Home_007_1", "Home_013_1", "Home_016_1", "Home_003_1", "Home_005_1", "Home_008_1","Home_001_1","Home_014_1","Home_014_2","Office_001_1"]
scene_list=["Home_003_2", "Home_005_2", "Home_010_1", "Home_001_2", "Home_004_1", "Home_006_1", "Home_011_1", "Home_015_1", "Home_002_1", "Home_004_2", "Home_007_1", "Home_013_1", "Home_016_1", "Home_003_1", "Home_005_1", "Home_008_1","Home_001_1","Home_014_1","Home_014_2","Office_001_1"]


HOME_DIR='/media/david/HardDrive/Documents/ActiveVisionDataset/'
DOWNSAMPLE_DIR='/media/david/HardDrive/Documents/ActiveVisionDataset_downsampled/'

INSTANCE_ID_MAP={
    "background" : 0,
    "advil_liqui_gels" : 1,
    "aunt_jemima_original_syrup" : 2,
    "bumblebee_albacore" : 3,
    "cholula_chipotle_hot_sauce" : 4,
    "coca_cola_glass_bottle" : 5,
    "crest_complete_minty_fresh" : 6,
    "crystal_hot_sauce" : 7,
    "expo_marker_red" : 8,
    "hersheys_bar" : 9,
    "honey_bunches_of_oats_honey_roasted" : 10,
    "honey_bunches_of_oats_with_almonds" : 11,
    "hunts_sauce" : 12,
    "listerine_green" : 13,
    "mahatma_rice" : 14,
    "nature_valley_granola_thins_dark_chocolate" : 15,
    "nutrigrain_harvest_blueberry_bliss" : 16,
    "pepto_bismol" : 17,
    "pringles_bbq" : 18,
    "progresso_new_england_clam_chowder" : 19,
    "quaker_chewy_low_fat_chocolate_chunk" : 20,
    "red_bull" : 21,
    "softsoap_clear" : 22,
    "softsoap_gold" : 23,
    "softsoap_white" : 24,
    "spongebob_squarepants_fruit_snaks" : 25,
    "tapatio_hot_sauce" : 26,
    "vo5_tea_therapy_healthful_green_tea_smoothing_shampoo" : 27,
    "nature_valley_sweet_and_salty_nut_almond" : 28,
    "nature_valley_sweet_and_salty_nut_cashew" : 29,
    "nature_valley_sweet_and_salty_nut_peanut" : 30,
    "nature_valley_sweet_and_salty_nut_roasted_mix_nut" : 31,
    "paper_plate" : 32,
    "red_cup" : 33
}

for scene in scene_list:
	max_box_areas[scene]={}
	scene_path = os.path.join(HOME_DIR,scene)
	images_path = os.path.join(scene_path,'jpg_rgb')
	depth_path = os.path.join(scene_path,'high_res_depth')
	annotations_path = os.path.join(scene_path,'annotations.json')

	instance_names_path = os.path.join(scene_path,'present_instance_names.txt')
    instance_file = open(instance_names_path)
    instance_names = instance_file.read().splitlines()

	#load data
	image_names = [f for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path,f))]
	image_names.sort()
	depth_names = [f for f in os.listdir(depth_path) if os.path.isfile(os.path.join(depth_path,f))]
	depth_names.sort()
	ann_file = open(annotations_path)
	annotations = json.load(ann_file)

	planner=Astar_planner(annotations,)

	for image in image_names:
		expert_moves={}

		for target_name in instance_names:
			target_id=INSTANCE_ID_MAP[target_name]

			#plan a path from the current image to the best target image
			first_move=get_first_move_in_plan(image,target_id)
			expert_moves[target_id]=first_move

		annotations[image]['expert']=expert_moves

	with open(os.path.join(DOWNSAMPLE_DIR,scene,'annotations.json'), 'w') as fp:
		json.dump(annotations,fp)

	copyfile(os.path.join(HOME_DIR,scene,'present_instance_names.txt'),os.path.join(DOWNSAMPLE_DIR,scene,'present_instance_names.txt'))