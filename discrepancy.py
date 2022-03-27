##############################################################################
# computing a receptive field for a specific unit, based on a specific image
##############################################################################
import sys
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import os
import datetime
import helpers

# renaming imported function for code legibility
network_forward_pass = helpers.network_forward_pass

# constants: script parameters
# LAYER, UNIT are used to target the unit we want to analyze
# PATH indicates the path to the dataset
LAYER = int(sys.argv[1])
UNIT = int(sys.argv[2])
NAME = sys.argv[3]
PATH = f"occluded/{NAME}"
FILE = f"discrepancy_maps/{NAME}.jpg"

activation_log = {}

# timestamp: exec clock starts
start = datetime.datetime.now()

start_message = f"Computing discrepancy map for image {NAME} using unit "
start_message += f"{UNIT} of layer {LAYER}\n-- Loading model:"
print(start_message)

model = torch.hub.load(
    'pytorch/vision:v0.10.0',
    'alexnet',
    pretrained=True
)


def activation_map(module, input, output,
    target_unit=UNIT):
    """
    note: the reference to output[0] assumes the batch submitted to the model
    only contains one image at a time 
    """
    activation_map = output[0][target_unit]
    activation_log.update({filename: activation_map})


# hook registration for target layer
model.features[LAYER].register_forward_hook(activation_map)

##############################################################################
# step 1
print("\n-- Passing baseline image to the network")
filename = f"{FILE}"
network_forward_pass(model, filename)

##############################################################################
# step 2
file_list = os.listdir(PATH)
print(f"\n-- Passing occluded images to the network ({len(file_list)} files)")

for file in file_list:
    filename = f"{PATH}\\{file}"
    network_forward_pass(model, filename)

    if file_list.index(file) == int(len(file_list) / 2):
        halftime = datetime.datetime.now()
        time = str((halftime - start).total_seconds())
        message = f"\nStill working (halfway through ; {time} seconds since"
        message += " execution started)"
        print(message)

##############################################################################
# step 3
print("\n-- Analyzing discrepancies...")

x_size, y_size = Image.open(FILE).size
output_image = Image.new("RGB", (x_size, y_size))

baseline = activation_log[FILE]
activation_log.pop(FILE)

for file in activation_log:
    activation = activation_log[file]
    difference = baseline - activation
    file = file.split("-")
    x_step = int(file[0].split("_")[-1])
    y_step = int(file[1].split(".")[0])
    value = int(torch.max(difference))
    #print(f"{x_step}, {y_step} : {value}")
    greyscale_shade = (10 * value, 10 * value, 10 * value)
    x_pos = x_step * 3
    y_pos = y_step * 3
    for x in range(x_pos, x_pos + 11):
        for y in range(y_pos, y_pos + 11):
            output_image.putpixel((x, y), greyscale_shade)

output_image.save(f"discrepancy_maps/{NAME}_discrepancy_l{LAYER}u{UNIT}.jpg", "jpeg")

# timestamp: exec clock stops
stop = datetime.datetime.now()    
timestamp = stop.strftime("%m%d_%H%M%S")
execution_time = str((stop - start).total_seconds())
end_message = f"\nDiscrepancy map created for image {NAME}."
end_message +=  f" Time: {execution_time} seconds"
print(end_message)