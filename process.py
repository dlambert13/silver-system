import helpers
import sys
import os

LAYER = sys.argv[1]
UNIT = sys.argv[2]
filename = "./logs/avatarnet_max_activation_log.txt"

image_list = helpers.top_10(LAYER, UNIT, filename)

if "occluded" not in os.listdir("."):
    os.mkdir("./occluded")

# for hub alexnet : resize
"""
for filename in image_list:
    helpers.resize(filename)

resized_files_list = []
for filename in image_list:
    buffer = filename.split("/")[-1]
    name = f"discrepancy_maps/0_base/resized_{buffer}"
    print(name)
    resized_files_list.append(name)

list_size = len(resized_files_list)

for i in range(list_size):
    print(f"Creating occluded images for file {i + 1} out of {list_size}")
    file = resized_files_list[i]
    helpers.occlusion(file)
"""

#otherwise
"""
list_size = len(image_list)

for i in range(list_size):
    print(f"Creating occluded images for file {i + 1} out of {list_size}")
    file = image_list[i]
    helpers.occlusion(file, stride=11)
"""

#top-1
#"""
file = image_list[0]
# WARNING: as of 0520, must match stride in discrepancy.py. TODO: stride as argv
helpers.occlusion(file, stride=5)
#"""
