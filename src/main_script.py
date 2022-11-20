"""Wraps image occlusion, discrepancy map computation, top 10 assembly and
receptive field computation in a single script for ease of use
Uses a pre-computed log file, stored in ../logs/
"""
import os
from helpers import top_10

# as a reference regarding the AlexNet architecture, the indices of the
# model's ReLU layers, and the associated number of units:
# l_u_dict = {1: 64, 4: 192, 7: 384, 9: 256, 11: 256}
LAYERS = [11] # layers to target, as a Python list
UNITS = [6] # units to target, as a Python list

# model selection : AvatarNet or default AlexNet
#MODEL_ID = "avn"
MODEL_ID = "axn"

# recommended values for the stride of the occlusion process to keep the
# execution time reasonable
# warning: in the rest of the pipeline, occluder size is equal to stride
if MODEL_ID == "avn":
    STRIDE = 10
else:
    STRIDE = 5

# to override the recommended values, uncomment the next line:
#STRIDE = 4

# retrieving log file
project_dir = os.getcwd()
log = os.path.join(
    project_dir,
    "logs",
    f"{MODEL_ID}_max_activation_log.txt"
)

# creating "results" directory where the script results will be stored
results_dir = os.path.join(project_dir, "results")
if "results" not in os.listdir(project_dir):
    os.mkdir(results_dir)

# loop over the selected layers and units
for l in LAYERS:
    for u in UNITS:
        #os.system(f"python ./src/occlusion.py {l} {u} {log} {STRIDE}")
        os.system(f"python ./src/discrepancy.py {l} {u} {log} {STRIDE}")
        os.system(f"python ./src/top10.py {l} {u} {log}")
        os.system(f"python ./src/rec_field.py {l} {u} {log}")
