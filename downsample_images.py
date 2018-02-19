import json
import os
import cv2 as cv
from shutil import copyfile
import numpy as np

#scene_list=["Home_003_2", "Home_005_2", "Home_010_1", "Home_001_2", "Home_004_1", "Home_006_1", "Home_011_1", "Home_015_1", "Home_002_1", "Home_004_2", "Home_007_1", "Home_013_1", "Home_016_1", "Home_003_1", "Home_005_1", "Home_008_1","Home_001_1","Home_014_1","Home_014_2","Office_001_1"]
scene_list=["Home_013_1", "Home_016_1", "Home_003_1", "Home_005_1", "Home_008_1","Home_001_1","Home_014_1","Home_014_2","Office_001_1"]
#scene_list=["Home_013_1", "Home_016_1", "Home_003_1", "Home_005_1", "Home_008_1","Home_001_1","Home_014_1","Home_014_2","Office_001_1"]

HOME_DIR=os.path.join(os.path.dirname(__file__),'ActiveVisionDataset/')
DOWNSAMPLE_DIR=os.path.join(os.path.dirname(__file__),'ActiveVisionDataset_downsampled/')

max_box_areas={}

resize_factor=0.5
screen_height=1080
screen_width=1920
new_shape=(224,224)

for scene in scene_list:
	max_box_areas[scene]={}
	scene_path = os.path.join(HOME_DIR,scene)
	images_path = os.path.join(scene_path,'jpg_rgb')
	depth_path = os.path.join(scene_path,'high_res_depth')
	annotations_path = os.path.join(scene_path,'annotations.json')

	#load data
	image_names = [f for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path,f))]
	image_names.sort()
	depth_names = [f for f in os.listdir(depth_path) if os.path.isfile(os.path.join(depth_path,f))]
	depth_names.sort()
	ann_file = open(annotations_path)
	annotations = json.load(ann_file)

	for image in image_names:
		print(scene,image)
		
		img=cv.imread(os.path.join(images_path,image))
		img=img[0:1080,420:1500,:]
		img=cv.resize(img,new_shape)

		depth_name=image[:-5]+'3.png'
		depth=cv.imread(os.path.join(depth_path,depth_name))
		depth=depth[0:1080,420:1500,:]
		#print(os.path.join(depth_path,depth_name))
		depth=cv.resize(depth,new_shape)
		new_image=image[:-3]+'png'
		combined_image=np.concatenate((img,depth[:,:,0:1]),axis=-1).astype(np.uint8)
		cv.imwrite(os.path.join(DOWNSAMPLE_DIR,scene,'jpg_rgb',new_image),combined_image)

		boxes=annotations[image]['bounding_boxes']
		for i in range(len(boxes)):
			for j in range(4):
				boxes[i][j]=int(round(boxes[i][j]*resize_factor))
		annotations[image]['bounding_boxes']=boxes
		annotations[image]['forward']=annotations[image]['forward'].replace('jpg','png')
		annotations[image]['rotate_ccw']=annotations[image]['rotate_ccw'].replace('jpg','png')
		annotations[image]['rotate_cw']=annotations[image]['rotate_cw'].replace('jpg','png')
		annotations[image]['left']=annotations[image]['left'].replace('jpg','png')
		annotations[image]['right']=annotations[image]['right'].replace('jpg','png')
		annotations[image]['backward']=annotations[image]['backward'].replace('jpg','png')

		annotations[new_image] = annotations.pop(image)

	with open(os.path.join(DOWNSAMPLE_DIR,scene,'annotations.json'), 'w') as fp:
		json.dump(annotations,fp)

	copyfile(os.path.join(HOME_DIR,scene,'present_instance_names.txt'),os.path.join(DOWNSAMPLE_DIR,scene,'present_instance_names.txt'))