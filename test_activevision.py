import gym
import gym_activevision
import random

env=gym.make('ActiveVision-v0')
env.reset()
env.render()

move_command = ''
done=False
while (move_command != 'q' and done is False):
	#get input from user 
	move_command = input('Enter command: ')

	#get the next image name to display based on the 
	#user input, and the annotation.
	action=0
	if move_command == 'w':
		action=0
	elif move_command == 'a':
		action=1
	elif move_command == 's':
		action=2
	elif move_command == 'd':
		action=3
	elif move_command == 'e':
		action=4
	elif move_command == 'r':
		action=5
	elif move_command == 'h':
		action=6
		print("{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n".format(
			  "Enter a character to move around the scene:",
			  "'w' - forward", 
			  "'a' - rotate counter clockwise", 
			  "'s' - backward", 
			  "'d' - rotate clockwise", 
			  "'e' - left", 
			  "'r' - right", 
			  "'q' - quit", 
			  "'h' - print this help menu"))
	_,reward,done,_=env.step(action)
	print(reward)
	env.render()