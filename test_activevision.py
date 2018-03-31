import gym
import gym_activevision
import random
import os
import json

env=gym.make('ActiveVision-v0')
info=env.reset()
env.render()

move_command = ''
done=False

HOME_DIR='/media/david/HardDrive/Documents/ActiveVisionDataset_downsampled/'
scene_path = os.path.join(HOME_DIR,info['scene'])
annotations_path = os.path.join(scene_path,'annotations.json')
ann_file = open(annotations_path)
annotations = json.load(ann_file)
target_id=info['target_id']

total_steps=100000
total_reward=0

# while (move_command != 'q'):
for i in range(total_steps):
	# #get input from user 
	# move_command = input('Enter command: ')

	# #get the next image name to display based on the 
	# #user input, and the annotation.
	# action=0
	# if move_command == 'w':
	# 	action=0
	# elif move_command == 'a':
	# 	action=1
	# elif move_command == 's':
	# 	action=2
	# elif move_command == 'd':
	# 	action=3
	# elif move_command == 'e':
	# 	action=4
	# elif move_command == 'r':
	# 	action=5
	# elif move_command == 'h':
	# 	action=6
	# 	print("{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n".format(
	# 		  "Enter a character to move around the scene:",
	# 		  "'w' - forward", 
	# 		  "'a' - rotate counter clockwise", 
	# 		  "'s' - backward", 
	# 		  "'d' - rotate clockwise", 
	# 		  "'e' - left", 
	# 		  "'r' - right", 
	# 		  "'q' - quit", 
	# 		  "'h' - print this help menu"))

	img=info['img_path'][-19:]
	action=annotations[img]['expert'][str(target_id)]
	_,reward,done,info=env.step(action)
	# print(reward,action)
	# env.render()
	total_reward+=reward
	print(total_reward/(i+1))
	if done:
		info=env.reset()
		scene_path = os.path.join(HOME_DIR,info['scene'])
		annotations_path = os.path.join(scene_path,'annotations.json')
		ann_file = open(annotations_path)
		annotations = json.load(ann_file)
		target_id=info['target_id']