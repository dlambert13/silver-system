import sys
import os
import helpers

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
base_dir = ".."

dc_dir = os.path.join(
    "..",
    "discrepancy",
    MODEL_ID,
    target_unit_id
)

rf_dir = os.path.join("..", "rf")
model_rf_dir = os.path.join(rf_dir, MODEL_ID)
unit_rf_dir = os.path.join(model_rf_dir, target_unit_id)

if "rf" not in os.listdir(base_dir):
    os.mkdir(rf_dir)

if MODEL_ID not in os.listdir(rf_dir):
    os.mkdir(model_rf_dir)

if target_unit_id not in os.listdir(model_rf_dir):
    os.mkdir(unit_rf_dir)

##############################################################################

base_list = []
max_filepaths = []
cal_filepaths = []

for filename in os.listdir(dc_dir):
    dc_filepath = os.path.join(dc_dir, filename) # points to current dc map
    base_filename = filename.split(".")[0] # removing file extension
    base_list.append(base_filename) # to automate the rest of the process
    
    # -- contour containing max activation in each dc map is saved to an aux file
    max_filepath = os.path.join(
        unit_rf_dir,
        f"{base_filename}_max.png"
    )
    max_filepaths.append(max_filepath) # to automate the rest of the process
    helpers.max_extractor(dc_filepath, max_filepath)
    
    # -- calibrating (centering) each contour and saving to a second aux file
    cal_filepath = os.path.join(
        unit_rf_dir,
        f"{base_filename}_cal.png"
    )
    cal_filepaths.append(cal_filepath) # to automate the rest of the process
    helpers.calibrate(max_filepath, cal_filepath)

rf_filepath = os.path.join(unit_rf_dir, f"{target_unit_id}_rf.png")
helpers.add(cal_filepaths, rf_filepath)

# cleaning up auxiliary files
#"""
for i in range(len(base_list)):
    os.remove(max_filepaths[i])
    os.remove(cal_filepaths[i])
#"""
