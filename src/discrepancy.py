"""Computing the top-10 discrepancy maps for a given unit on a given dataset,
using the occluded images resulting from occlusion.py
"""

##############################################################################
import os
import sys
import torch
from PIL import Image
import datetime
from helpers import network_forward_pass, top_10

##############################################################################
# constants and parameters (CLI args or otherwise)
##############################################################################
# toggles GPU usage; placeholder for future GPU usage implementation
GPU = False

# constants: script parameters
# LAYER, UNIT are used to target the unit we want to analyze
# LOG indicates the log to use
LAYER = int(sys.argv[1])
UNIT = int(sys.argv[2])
LOG = sys.argv[3]

# target unit has a unique identifier, used in directory and file creation
target_unit_id = f"l{LAYER}u{UNIT}"

# retrieving model id from log filename:
# os.path.split returns (head, tail) ; filename is tail
_, log_filename = os.path.split(LOG)
# model id is before the first underscore
MODEL_ID = log_filename.split('_')[0]

# STRIDE and OCCLUDER have a direct impact on the number of files to process;
# higher values can be used to speed up the process, but the readability of
# the discrepancy maps will drop accordingly
STRIDE = int(sys.argv[4])
OCCLUDER = STRIDE
# FACTOR is the multiplier for the value of the difference in activation
# between the baseline image and the occluded image; higher values make
# discrepancy maps more legible; no effect on computation time
FACTOR = 15

##############################################################################
# directory structure creation/verification
##############################################################################
base_dir = os.path.join(os.getcwd(), "results")  # base results directory
# occluded images directory (output of occlusion.py)
occluded_base_dir = os.path.join(base_dir, "tmp")
# directory where the model-specific results will be saved
model_dir = os.path.join(base_dir, MODEL_ID)
# discrepancy maps directory within the model-specific directory
dm_dir = os.path.join(model_dir, "discrepancy")
# directory where the unit-specific discrepancy maps will be saved
unit_dm_dir = os.path.join(dm_dir, target_unit_id)

if MODEL_ID not in os.listdir(base_dir):
    os.mkdir(model_dir)
if "discrepancy" not in os.listdir(model_dir):
    os.mkdir(dm_dir)
if target_unit_id not in os.listdir(dm_dir):
    os.mkdir(unit_dm_dir)

init_message = (
    "\n------> Computing discrepancy maps\n"
    f"Model-specific directory: {model_dir}\n"
    f"Results for this unit will be saved in: {unit_dm_dir}"
)
print(init_message)

##############################################################################
# retrieving top-10 file names from log and loading the network
##############################################################################
image_list = top_10(LAYER, UNIT, LOG)
nb_images = len(image_list)

if MODEL_ID == "avn":
    streetlab_model_file = os.path.join(
        base_dir,
        "avatar/alexnet_places_old.pt"
    )
    model = torch.load(
        streetlab_model_file,
        map_location=torch.device('cpu')
    )
else:
    print("No model specified: defaulting to AlexNet")
    # loading the model
    model = torch.hub.load(
        'pytorch/vision:v0.10.0',
        'alexnet',
        pretrained=True
    )
    # pointing to the resized images
    resized_dir = os.path.join(occluded_base_dir, "resized_axn")
    buffer_image_list = []
    for top10_filepath in image_list:
        _, top10_filename = os.path.split(top10_filepath)
        resized_filepath = os.path.join(resized_dir, top10_filename)
        buffer_image_list.append(resized_filepath)
    image_list = buffer_image_list

##############################################################################
# looping over the 10 files:
# step 1: baseline forward pass, 2: occluded images forward pass,
# and 3: discrepancy analysis to compute the discrepancy map
##############################################################################
for image_filepath in image_list:
    _, image_filename = os.path.split(image_filepath)
    # removing the file extension: splitting by "." character
    image_name = image_filename.split(".")[0]
    # locating the occluded images corresponding to the file from the top 10
    occluded_images_dir = os.path.join(occluded_base_dir, image_name)

    activation_log = {}

    # timestamp: exec clock starts
    start = datetime.datetime.now()

    current_index = image_list.index(image_filepath) + 1
    start_template = (
        "\n---- Processing image {} out of {}\n"
        "Computing discrepancy map for image \'{}\'"
        "using unit {} of layer {}\n"
        "-- Loading model"
    )

    start_message = start_template.format(
        current_index,
        nb_images,
        image_filename,
        UNIT,
        LAYER
    )
    print(start_message)

    def activation_map(module, input, output, target_unit=UNIT):
        activation_map = output[0][target_unit]
        activation_log.update({current_filepath: activation_map})

    # hook registration for target layer
    model.features[LAYER].register_forward_hook(activation_map)

    ##########################################################################
    # step 1
    print("-- Passing baseline image to the network")
    # network_forward_pass triggers the forward hook which logs the activation
    # the dummy output of network_forward_pass is not used in this context
    current_filepath = image_filepath
    if MODEL_ID == "avn":
        network_forward_pass(model, current_filepath, mode="avatar", gpu=GPU)
    else:
        network_forward_pass(model, current_filepath, gpu=GPU)

    ##########################################################################
    # step 2: passing the occluded images through the network
    ##########################################################################
    occluded_list = os.listdir(occluded_images_dir)
    occl_template = "-- Passing occluded images to the network ({} files)"
    print(occl_template.format(len(occluded_list)))

    occluded_list.sort()

    for occluded_im_filename in occluded_list:
        occluded_im_filepath = os.path.join(
            occluded_images_dir,
            occluded_im_filename
        )

        current_filepath = occluded_im_filepath
        # network_forward_pass triggers the forward hook, logs the activation
        # the dummy output of network_forward_pass is not used in this context
        if MODEL_ID == "avn":
            network_forward_pass(
                model,
                current_filepath,
                mode="avatar", gpu=GPU
            )
        else:
            network_forward_pass(
                model,
                current_filepath,
                gpu=GPU
            )

        current_index = occluded_list.index(occluded_im_filename)
        list_half_index = int(len(occluded_list) / 2)

        if current_index == list_half_index:
            halftime = datetime.datetime.now()
            time = (halftime - start).total_seconds()
            message = (
                "\tStill working (halfway through ; {:.2f} seconds"
                " since execution started)"
            )
            print(message.format(time))

    ##########################################################################
    # step 3
    ##########################################################################
    print("-- Analyzing discrepancies...")

    x_size, y_size = Image.open(image_filepath).size
    output_image = Image.new("RGB", (x_size, y_size))

    # storing the baseline activation in a variable
    baseline = activation_log[image_filepath]
    # removing baseline filepath from activation log
    activation_log.pop(image_filepath)

    # each log entry is the filepath to an occluded image
    # occluded filepath format is "..._x-y.png", x and y are the coordinates
    # we retrieve the coordinates of the occluder from the filepath
    for log_entry in activation_log:
        activation = activation_log[log_entry]
        difference = baseline - activation
        log_entry = log_entry.split("_")[-1]
        log_entry = log_entry.split("-")
        x_step = int(log_entry[0])
        y_step = int(log_entry[1].rstrip(".png"))
        value = int(FACTOR * torch.max(difference))
        greyscale_shade = (value, value, value)
        x_pos = x_step * STRIDE
        y_pos = y_step * STRIDE
        for x in range(x_pos, x_pos + OCCLUDER):
            for y in range(y_pos, y_pos + OCCLUDER):
                output_image.putpixel((x, y), greyscale_shade)

    out_filename = os.path.join(
        unit_dm_dir,
        f"{image_name}_dc{STRIDE}_{target_unit_id}.png"
    )
    output_image.save(out_filename, "png")

    # timestamp: exec clock stops
    stop = datetime.datetime.now()
    timestamp = stop.strftime("%m%d_%H%M%S")
    execution_time = (stop - start).total_seconds()
    end_template = (
        "Discrepancy map created for image {}.\n"
        "Time: {:.2f} seconds\n"
    )
    end_message = end_template.format(image_name, execution_time)
    print(end_message)
