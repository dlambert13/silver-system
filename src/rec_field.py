"""Computes the unit's receptive field from the discrepancy maps obtained
from the top 10 images
"""

##############################################################################
import sys
import os
from helpers import max_extractor, calibrate, add

##############################################################################
# constants and parameters (CLI args or otherwise)
##############################################################################
LAYER = sys.argv[1]
UNIT = sys.argv[2]
LOG = sys.argv[3]

target_unit_id = f"l{LAYER}u{UNIT}"

# fetching log filename from log path to retrieve model id
# os.path.split returns (head, tail) ; filename is tail
_, log_filename = os.path.split(LOG)
# model id is before the first underscore
MODEL_ID = log_filename.split('_')[0]

##############################################################################
# directory structure creation/verification and save path
##############################################################################
base_dir = os.path.join(os.getcwd(), "results")  # base results directory
model_dir = os.path.join(base_dir, MODEL_ID)  # model-specific directory
dc_dir = os.path.join(
    base_dir,
    MODEL_ID,
    "discrepancy",
    target_unit_id
)
rf_dir = os.path.join(model_dir, "rf")
rf_filepath = os.path.join(rf_dir, f"{target_unit_id}_rf.png")

if MODEL_ID not in os.listdir(base_dir):
    os.mkdir(model_dir)

if "rf" not in os.listdir(model_dir):
    os.mkdir(rf_dir)

init_message = (
    "\n------> Computing receptive field from top 10 discrepancy maps\n"
    f"Model-specific directory: {model_dir}\n"
    f"Results for this unit will be saved in: {rf_dir}"
)
print(init_message)

##############################################################################
# receptive field computation from the top 10 discrepancy maps
##############################################################################
# creating lists where filenames are stored
# to automate the cleanup process (see the bottom of the script)
base_list = []
max_filepaths = []
cal_filepaths = []

for filename in os.listdir(dc_dir):
    dc_filepath = os.path.join(dc_dir, filename)  # points to current dc map
    base_filename = filename.split(".")[0]  # removing file extension
    base_list.append(base_filename)  # storing base filename for later lookup

    # contour containing max activation in each dc map is saved to an aux file
    max_filepath = os.path.join(
        rf_dir,
        f"{base_filename}_max.png"
    )
    max_filepaths.append(max_filepath)  # to automate cleanup
    max_extractor(dc_filepath, max_filepath)  # see helpers.max_extractor

    # calibrating (centering) each contour and saving to a second aux file
    cal_filepath = os.path.join(
        rf_dir,
        f"{base_filename}_cal.png"
    )
    cal_filepaths.append(cal_filepath)  # to automate the rest of the process
    calibrate(max_filepath, cal_filepath)  # see helpers.calibrate
# overlaying the ten calibrated images together; see helpers.add
add(cal_filepaths, rf_filepath)

##############################################################################
# cleaning up auxiliary files
##############################################################################
# comment out to keep the auxiliary files
# """
for i in range(len(base_list)):
    os.remove(max_filepaths[i])
    os.remove(cal_filepaths[i])
# """
