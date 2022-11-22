"""Assembles:
- the top 10 images;
- the associated discrepancy maps;
- the discrepancy maps overlaid on top of the corresponding images.
"""

##############################################################################
from helpers import top_10, stack, mask_overlay
import sys
import os

##############################################################################
# constants and parameters (CLI args or otherwise)
##############################################################################
LAYER = sys.argv[1]
UNIT = sys.argv[2]
LOG = sys.argv[3]
# fetching log filename from log path to retrieve model id
# os.path.split returns (head, tail) ; filename is tail
_, log_filename = os.path.split(LOG)
# model id is before the first underscore
MODEL_ID = log_filename.split('_')[0]
# target_unit_id combines the layer and unit index
target_unit_id = f"l{LAYER}u{UNIT}"
base_filename = "top10_" + target_unit_id

##############################################################################
# directory structure creation/verification and save paths
##############################################################################
base_dir = os.path.join(os.getcwd(), "results") # base results directory
model_dir = os.path.join(base_dir, MODEL_ID) # model-specific directory
save_dir = os.path.join(model_dir, "top10") # top 10 storage directory
dc_dir = os.path.join(model_dir, "discrepancy", target_unit_id)

# save path for the assembled top-10 images
save_path = os.path.join(save_dir, f"{base_filename}.png")
# save path for the assembled discrepancy maps
dc_save_path = os.path.join(save_dir, f"{base_filename}_dc.png")
# save path for the overlaid images
ol_save_path = os.path.join(save_dir, f"{base_filename}_ol.png")
# save path for the final image stacking the three images above
stack_save_path = os.path.join(save_dir, f"{base_filename}_stack.png")

# directory structure creation if it doesn't exist yet
# top10 directory, with a subdir for each model
if "top10" not in os.listdir(model_dir):
    os.mkdir(save_dir)

init_message = (
    "\n------> Assembling top ten images and their discrepancy maps\n"
    f"Model-specific directory: {model_dir}\n"
    f"Results for this unit will be saved in: {save_dir}"
)
print(init_message)

##############################################################################
# assembling the top 10 images
##############################################################################
filepath_list = top_10(LAYER, UNIT, LOG)

if MODEL_ID == "axn":
    # pointing to the resized images
        resized_dir = os.path.join(base_dir, "tmp", "resized_axn")
        buffer_filepath_list = []
        for top10_filepath in filepath_list:
            _, top10_filename = os.path.split(top10_filepath)
            resized_filepath = os.path.join(resized_dir, top10_filename)
            buffer_filepath_list.append(resized_filepath)
        filepath_list = buffer_filepath_list

stack(filepath_list, save_path)

##############################################################################
# assembling the top 10 discrepancy maps
##############################################################################
# the trick here is to stack the files in the same order as they are in
# filepath_list: looking them up on disk will return them in alphabetical
# order, which might be different from their order in the top 10
# -- first, retrieve the stride indicator ("..._dcNN_...")
dummy_list = os.listdir(dc_dir)
_, dummy_filename = os.path.split(dummy_list[0])
stride_indicator = dummy_filename.split("_")[1]
# -- then, recreate the file paths based on their order in the top 10
dc_filepath_list = []
for i in range(len(filepath_list)):
    # retrieve file name
    _, dummy_filename = os.path.split(filepath_list[i])
    # remove file extension
    dummy_basename = dummy_filename.split(".")[0]
    # reconstruct filename by adding the remaining elements
    filename = dummy_basename + f"_{stride_indicator}_{target_unit_id}.png"
    # join with dc_dir and then add to dc_filepath_list
    dc_filepath_list += [
        os.path.join(
            dc_dir,
            filename
        )
    ]

stack(dc_filepath_list, dc_save_path)

##############################################################################
# overlaying discrepancy maps on top of the top 10 images
##############################################################################
mask_overlay(save_path, dc_save_path, ol_save_path)

##############################################################################
# stacking top 10 images, dc maps, and overlaid images for interpretation
##############################################################################
stack(
    [save_path, dc_save_path, ol_save_path],
    stack_save_path,
    direction="column"
)

##############################################################################
# cleaning up auxiliary files
##############################################################################
# comment out to keep the auxiliary files
#"""
os.remove(save_path)
os.remove(dc_save_path)
os.remove(ol_save_path)
#"""
