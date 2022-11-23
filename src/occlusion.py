"""Creating the occluded images from the top-10 images for a given unit, based
on the information retrieved from the pre-computed log file
"""

##############################################################################
import sys
import os
from helpers import top_10, resize, occlusion

##############################################################################
# constants and parameters (CLI args or otherwise)
##############################################################################
LAYER = sys.argv[1]
UNIT = sys.argv[2]
LOG = sys.argv[3]

# WARNING: stride here must match stride in discrepancy.py
STRIDE = int(sys.argv[4])
OCCLUDER = STRIDE

##############################################################################
# log filename retrieval and directory structure definition
##############################################################################
# retrieving model id from log filename:
# os.path.split returns (head, tail) ; filename is tail
_, log_filename = os.path.split(LOG)
# model id is before the first underscore
MODEL_ID = log_filename.split('_')[0]

base_dir = os.path.join(os.getcwd(), "results") # base results directory
temporary_dir = os.path.join(base_dir, "tmp")
if "tmp" not in os.listdir(base_dir):
    os.mkdir(temporary_dir)

##############################################################################
# top-10 retrieval and resized image creation if necessary
##############################################################################
image_list = top_10(LAYER, UNIT, LOG)

# -- for axn : the images must first be resized
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
        resize(filepath, save_filepath)
    
    # overwrite the result of top_10 with the resized images
    image_list = buffer_list

##############################################################################
# looping over the ten images to create the corresponding occluded images
##############################################################################
list_size = len(image_list)

for i in range(list_size):
    print(f"--- Creating occluded images for file {i + 1} out of {list_size}")
    image_filepath = image_list[i]
    occlusion(image_filepath, temporary_dir, stride=STRIDE, occluder=OCCLUDER)
