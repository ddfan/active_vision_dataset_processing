import json
import os

scene_list=["Home_003_2", "Home_005_2", "Home_010_1", "Home_001_2", "Home_004_1", "Home_006_1", "Home_011_1", "Home_015_1", "Home_002_1", "Home_004_2", "Home_007_1", "Home_013_1", "Home_016_1", "Home_003_1", "Home_005_1", "Home_008_1","Home_001_1","Home_014_1","Home_014_2","Office_001_1"]

HOME_DIR=os.path.join(os.path.dirname(__file__),'ActiveVisionDataset/')

max_box_areas={}

for scene in scene_list:
	max_box_areas[scene]={}
	scene_path = os.path.join(HOME_DIR,scene)
	images_path = os.path.join(scene_path,'jpg_rgb')
	annotations_path = os.path.join(scene_path,'annotations.json')

	#load data
	image_names = os.listdir(images_path)
	image_names.sort()
	ann_file = open(annotations_path)
	annotations = json.load(ann_file)

	for image in image_names:
		boxes=annotations[image]['bounding_boxes']
		for box in boxes:
			box_size=(box[2]-box[0])*(box[3]-box[1])
			if box[4] in max_box_areas[scene]:
				if box_size>max_box_areas[scene][box[4]]:
					max_box_areas[scene][box[4]]=box_size
			else:
				max_box_areas[scene][box[4]]=box_size



with open('max_box_areas.json', 'w') as fp:
	json.dump(max_box_areas,fp)