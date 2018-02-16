import os
import sys
import visualizations.visualizations as vis


ROHIT_BASE_PATH = '/home/aeuser/Documents/active_vision_dataset_processing/ActiveVisionDataset/'


function_name = sys.argv[1]
scene_name = sys.argv[2]
scene_path = os.path.join(ROHIT_BASE_PATH,scene_name)

if function_name == 'vis_boxes_and_move':
    vis.vis_boxes_and_move(scene_path)
elif function_name == 'vis_camera_pos_dirs':
    vis.vis_camera_pos_dirs(scene_path)
