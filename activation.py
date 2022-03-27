"""
TODO:
write docstring
write comments
save top LENGTH to .py file for automated reuse
"""

##############################################################################
# imports
##############################################################################
import sys
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
import os
import datetime
import helpers

##############################################################################
# constants: script parameters (command line arguments)
##############################################################################
# LAYER (argv[1]) and UNIT (argv[2]) are used to target the unit we want to
# analyze
# PATH (argv[3]) indicates the path to the dataset folder, as a subfolder of
# the ".\datasets" folder
# LENGTH defaults to 5
# ACTIVATION_MEASUREMENT, optional, can specify the function we use
# to measure the activation of the unit; defaults to mean if no specification 
LAYER = int(sys.argv[1])
UNIT = int(sys.argv[2])
PATH = ".\\datasets\\" + sys.argv[3]
if sys.argv[4] == "top10":
    LENGTH = 10
else:
    LENGTH = 5
if len(sys.argv) == 6 and sys.argv[5] == "max":
    ACTIVATION_MEASUREMENT = torch.max
    measurement = sys.argv[5]
else:
    ACTIVATION_MEASUREMENT = torch.mean
    measurement = "mean"

# ACTIVATION_LOG creates the dictionnary used to log the activation values
ACTIVATION_LOG = {}

##############################################################################
# functions
##############################################################################
def max_activation(module, input, output,
    measure=ACTIVATION_MEASUREMENT,
    log=ACTIVATION_LOG,
    target_unit=UNIT):
    """logs the activation of a unit to the specified logging dictionary
    - keys: filenames
    - values: activation values, defined using a measurement function passed
    as a keyword argument
    note: the reference to output[0] assumes the batch submitted to the model
    only contains one image at a time 
    """
    max_activation = float(measure(output[0][target_unit]))
    log.update({file: max_activation})

##############################################################################
# body
##############################################################################

# timestamp: exec clock starts
start = datetime.datetime.now()

# step 1
print("--- Loading model...")
model = torch.hub.load(
    'pytorch/vision:v0.10.0',
    'alexnet',
    pretrained=True
)

# step 2: forward hook registration for target layer
# the forward hook is the function that allows us to log the activation of a
# specific unit; see def max_activation above
model.features[LAYER].register_forward_hook(max_activation)

# step 3: passing images to the network
file_list = os.listdir(PATH)
print(f"--- Going through file list ({len(file_list)} files)...")
for file in file_list:
    filename = f"{PATH}\\{file}"
    helpers.network_forward_pass(model, filename)

# step 4: sorting images
print("--- Sorting images...")
# inverse activation log : dictionary reversal to query the activation log by
# value and return a file id
inverse_activation_log = {value: key for key, value in ACTIVATION_LOG.items()}

# list and sort activation values for each image
activation_values = [ACTIVATION_LOG[item] for item in ACTIVATION_LOG]
activation_values.sort(reverse=True)

# step 5: top LENGTH images (LENGTH = 5 or 10, defaults to 5)
output_image = Image.new("RGB", (LENGTH * 256,256 + 50))
imgx, imgy = output_image.size

# text setup
draw = ImageDraw.Draw(output_image)
font_path = "C:\\Windows\\Fonts\\bahnschrift.ttf"
font_size = 12
font = ImageFont.truetype(font_path, font_size)

layer_desc = model.features[LAYER]
print(f"Top 5 images for unit {UNIT} of layer {LAYER} ({layer_desc}):")
#creating the white background for the text
for y in range (256, imgy):
    for x in range(imgx):
        output_image.putpixel((x, y), (255, 255, 255))

# at the top of the output image: the top LENGTH images
for position in range(LENGTH):
    value = activation_values[position]
    position_filename = inverse_activation_log[value]
    image_filepath = f"{PATH}\\{position_filename}"
    image = Image.open(image_filepath)
    output_image.paste(image, (position * 256, 0))
    draw.text(
        (position * 256, 256),
        position_filename,
        font=font,
        fill=(0,0,0)
    )
    print(position_filename)

# below: the description
description = f"top {LENGTH} images from dataset at \"{PATH}\""
description += f"\t unit {UNIT} of layer {LAYER} ({layer_desc})"
description += f"using {measurement}"
draw.text(
        (0, 256 + 25),
        description,
        font=font,
        fill=(0,0,0)
)

# step 6: logging to .py file
"""
LAYER
UNIT
PATH
LENGTH
measurement 
top {LENGTH}
"""

# timestamp: exec clock stops
stop = datetime.datetime.now()    
timestamp = stop.strftime("%m%d_%H%M%S")
savepath = f"top{LENGTH}{measurement}/l{LAYER}u{UNIT}_{timestamp}.jpg"

output_image.save(savepath, "jpeg")

print(f"--- top {LENGTH} images successfully saved in file {savepath}")
execution_time = str((stop - start).total_seconds())
print(f"executed in {execution_time} seconds")
