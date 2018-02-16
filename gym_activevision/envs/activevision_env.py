import gym
from gym import error, spaces, utils
from gym.utils import seeding
import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import misc
import scipy.io as sio
import random
import cv2 as cv
import numpy as np

HOME_DIR=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),'ActiveVisionDataset/')

ACTION_MEANING = {
	0 : "w",
	1 : "a",
	2 : "s",
	3 : "d",
	4 : "l",
	5 : "r",
	6 : "n",
}

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

class ActiveVisionEnv(gym.Env):
	metadata = {'render.modes': ['human']}

	def __init__(self,training_mode_on=True, obs_type='img', episode_length=10):
		assert obs_type in ('img', 'state')
		self.scene_list=[]
		if training_mode_on:
			self.scene_list=["Home_003_2", "Home_005_2", "Home_010_1", "Home_001_2", "Home_004_1", "Home_006_1", "Home_011_1", "Home_015_1", "Home_002_1", "Home_004_2", "Home_007_1", "Home_013_1", "Home_016_1", "Home_003_1", "Home_005_1", "Home_008_1"]
		else:
			self.scene_list=["Home_001_1","Home_014_1","Home_014_2"]

		#load max box areas for normalizing rewards
		max_box_areas_file=open(os.path.join(HOME_DIR,"max_box_areas.json"))
		self.max_box_areas=json.load(max_box_areas_file)
		self.episode_length=episode_length
		self._obs_type=obs_type
		self.viewer = None
		self.all_boxes=[]
		self.target_boxes=[]
		self.action_space = spaces.Discrete(7)
		
		screen_height=1080
		screen_width=1920
		if self._obs_type == 'state':
			raise
			self.observation_space = spaces.Box(low=0, high=255, dtype=np.uint8, shape=(128,))
		elif self._obs_type == 'img':
			self.observation_space = spaces.Box(low=0, high=255, shape=(screen_height, screen_width, 3), dtype=np.uint8)
		else:
			raise error.Error('Unrecognized observation type: {}'.format(self._obs_type))


		self._random_init()

	def step(self, action):
		"""

		Parameters
		----------
		action :

		Returns
		-------
		ob, reward, episode_over, info : tuple
		"""

		self.move_command=ACTION_MEANING[action]
		if self.move_command == 'w':
			self.next_image_name = self.annotations[self.cur_image_name]['forward']
		elif self.move_command == 'a':
			self.next_image_name = self.annotations[self.cur_image_name]['rotate_ccw']
		elif self.move_command == 's':
			self.next_image_name = self.annotations[self.cur_image_name]['backward']
		elif self.move_command == 'd':
			self.next_image_name = self.annotations[self.cur_image_name]['rotate_cw']
		elif self.move_command == 'l':
			self.next_image_name = self.annotations[self.cur_image_name]['left']
		elif self.move_command == 'r':
			self.next_image_name = self.annotations[self.cur_image_name]['right']
		elif self.move_command == 'n':
			self.next_image_name = self.cur_image_name

		#If the move was not valid, the current image will be displayed again
		if self.next_image_name != '':
			self.cur_image_name = self.next_image_name

		#update boxes
		self.all_boxes = self.annotations[self.cur_image_name]['bounding_boxes']
		self.target_boxes=[]
		for box in self.all_boxes:
			if box[4] is self.target_id:
				self.target_boxes.append(box)
		
		#update reward, observations, time, check termination
		self.t+=1
		reward=self._get_reward()
		#print(self.t,action, reward)
		return self._get_obs(), reward, self._check_termination(), {"target_id":self.target_id}

	# return: (states, observations)
	def reset(self):
		self.t=0
		self.reward=0

		self._random_init()
		#return self._get_obs(),self.target_id
		return self._get_obs()

	def render(self, mode='human'):
		img = self._get_image()
		#draw red bounding boxes for all objects
		for box in self.all_boxes:
			cv.rectangle(img,(box[0],box[1]),(box[2],box[3]),(255,0,0),2)

		#draw green bounding boxes for target object(s)
		for box in self.target_boxes:
			cv.rectangle(img,(box[0],box[1]),(box[2],box[3]),(0,255,0),2)

		#return or render
		if mode == 'rgb_array':
			return img
		elif mode == 'human':
			from gym.envs.classic_control import rendering
			if self.viewer is None:
				self.viewer = rendering.SimpleImageViewer()
			self.viewer.imshow(img)
			return self.viewer.isopen

	def close(self):
		if self.viewer is not None:
			self.viewer.close()
			self.viewer = None
			
	def _get_image(self):
		#load the current image
		return misc.imread(os.path.join(self.images_path,self.cur_image_name))

	def _get_state(self):
		return self.state_dict

	def _get_obs(self):
		if self._obs_type == 'img':
			return self._get_image()
		if self._obs_type == 'state':
			return self._get_state()

	def _random_init(self):
		#pick a random scene
		self.scene=random.choice(self.scene_list)
		scene_path = os.path.join(HOME_DIR,self.scene)
		self.images_path = os.path.join(scene_path,'jpg_rgb')
		annotations_path = os.path.join(scene_path,'annotations.json')

		#set up scene specfic paths
		self.images_path = os.path.join(scene_path,'jpg_rgb')
		annotations_path = os.path.join(scene_path,'annotations.json')

		#load data
		self.image_names = os.listdir(self.images_path)
		self.image_names.sort()
		ann_file = open(annotations_path)
		self.annotations = json.load(ann_file)

		#set up for first image, pick a random starting location
		self.cur_image_name = random.choice(self.image_names)
		self.next_image_name = ''

		#pick a random object to look for
		#get list of objects in scene
		instance_names_path = os.path.join(scene_path,'present_instance_names.txt')
		instance_file = open(instance_names_path)
		instance_names = instance_file.read().splitlines() 
		self.target_id = INSTANCE_ID_MAP[random.choice(instance_names)]

	def _check_termination(self):
		if self.t >= self.episode_length:
			return True
		else:
			return False

	def _get_reward(self):
		this_reward=0
		#calculate area of box, then normalize by max box sizes
		for box in self.target_boxes:
			box_area=(box[2]-box[0])*(box[3]-box[1])
			this_reward+=float(box_area)/self.max_box_areas[self.scene][str(self.target_id)]
		return this_reward*10

