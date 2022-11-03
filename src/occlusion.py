"""
"""

import helpers
import sys
import os

LAYER = sys.argv[1]
UNIT = sys.argv[2]
LOG = sys.argv[3]

# WARNING: stride here must match stride in discrepancy.py
STRIDE = int(sys.argv[4])
OCCLUDER = STRIDE

# retrieving model id from log filename:
# os.path.split returns (head, tail) ; filename is tail
_, log_filename = os.path.split(LOG)
# model id is before the first underscore
MODEL_ID = log_filename.split('_')[0]

temporary_dir = os.path.join("..", "tmp")

if "tmp" not in os.listdir(".."):
    os.mkdir(temporary_dir)

##############################################################################
image_list = helpers.top_10(LAYER, UNIT, LOG)

# -- for hub alexnet : resize
if MODEL_ID == "axn":
    resized_dir = os.path.join(temporary_dir, "resized_axn")
    
    # creating dir if it does not yet exist
    if "resized_axn" not in os.listdir(temporary_dir):
        os.mkdir(resized_dir)
    
    # loop over the result of top_10, resizing each image
    buffer_list = []
    for filepath in image_list:
        _, filename = os.path.split(filepath)
        save_filepath = os.path.join(resized_dir, filename)
        buffer_list.append(save_filepath)
        # resizing and saving to save_filepath using default size (224, 224)
        helpers.resize(filepath, save_filepath)
    
    # overwrite the result of top_10 with the resized images
    image_list = buffer_list

# -- common to axn and avn
list_size = len(image_list)

for i in range(list_size):
    print(f"---- Creating occluded images for file {i + 1} out of {list_size}")
    image_file = image_list[i]
    helpers.occlusion(image_file, stride=STRIDE, occluder=OCCLUDER)
