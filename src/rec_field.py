import sys
import os
from helpers import max_extractor, calibrate, add

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
# directory structure creation/verification
##############################################################################
base_dir = os.path.join(os.getcwd(), "results") # base results directory
dc_dir = os.path.join(
    base_dir,
    "discrepancy",
    MODEL_ID,
    target_unit_id
)
rf_dir = os.path.join(base_dir, "rf")
model_rf_dir = os.path.join(rf_dir, MODEL_ID)
unit_rf_dir = os.path.join(model_rf_dir, target_unit_id)

if "rf" not in os.listdir(base_dir):
    os.mkdir(rf_dir)

if MODEL_ID not in os.listdir(rf_dir):
    os.mkdir(model_rf_dir)

if target_unit_id not in os.listdir(model_rf_dir):
    os.mkdir(unit_rf_dir)

##############################################################################
# receptive field computation from the top 10 discrepancy maps
##############################################################################
# creating lists where filenames are stored
# to automate the cleanup process (see the bottom of the script)
base_list = []
max_filepaths = []
cal_filepaths = []

for filename in os.listdir(dc_dir):
    dc_filepath = os.path.join(dc_dir, filename) # points to current dc map
    base_filename = filename.split(".")[0] # removing file extension
    base_list.append(base_filename) # storing base filename for later lookup
    
    # contour containing max activation in each dc map is saved to an aux file
    max_filepath = os.path.join(
        unit_rf_dir,
        f"{base_filename}_max.png"
    )
    max_filepaths.append(max_filepath) # to automate cleanup
    max_extractor(dc_filepath, max_filepath) # see helpers.max_extractor
    
    # calibrating (centering) each contour and saving to a second aux file
    cal_filepath = os.path.join(
        unit_rf_dir,
        f"{base_filename}_cal.png"
    )
    cal_filepaths.append(cal_filepath) # to automate the rest of the process
    calibrate(max_filepath, cal_filepath) # see helpers.calibrate

rf_filepath = os.path.join(unit_rf_dir, f"{target_unit_id}_rf.png")
 # overlaying the ten calibrated images together; see helpers.add
add(cal_filepaths, rf_filepath)

##############################################################################
# cleaning up auxiliary files
##############################################################################
# comment out to keep the auxiliary files
#"""
for i in range(len(base_list)):
    os.remove(max_filepaths[i])
    os.remove(cal_filepaths[i])
#"""
