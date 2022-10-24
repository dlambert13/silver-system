"""
---> README (short version)
>python activation.py test_dataset
will log the top 10 images for each unit of all ReLU layers, using max
activation to sort the images found at "./datasets/val256_color"

>python activation.py 1 23 val256_color mean
same but using mean activation to sort the images

--> DEPENDENCIES (must be in same folder as activation.py)
- helpers.py
- dataset folder, inside a "datasets" folder ; "flat" folder (all images
together; no subfolder)
"""

##############################################################################
# imports
##############################################################################
print("\n--- Importing dependencies...")
import sys
import torch
import os
import datetime
import helpers

##############################################################################
# constants: script parameters (command line arguments)
##############################################################################
# DATASET (sys.argv[1]) indicates the dataset folder, as a subfolder of
# the ".\datasets" folder
# ACTIVATION_MEASUREMENT, optional, can specify the function we use
# to measure the activation of the unit; defaults to max if not specified 
print("\n--- Parsing arguments...")
DATASET = sys.argv[1]
PATH = f".\\datasets\\{DATASET}"

if len(sys.argv) >= 3 and sys.argv[2] == "mean":
    ACTIVATION_MEASUREMENT = torch.mean
    MEASURE_STRING = sys.argv[2]
else:
    ACTIVATION_MEASUREMENT = torch.max
    MEASURE_STRING = "max"

file_list = os.listdir(PATH)
dataset_len = len(file_list)

# logging ReLU layers only
# the number of units in ReLU layers is hardcoded here
# log tensor row represent a unit, columns a file
LAYERS = [1, 4, 7, 9, 11]
UNITS = [64, 192, 384, 256, 256]
activation_log_1 = torch.as_tensor([[0] * dataset_len] * UNITS[0])
activation_log_4 = torch.as_tensor([[0] * dataset_len] * UNITS[1])
activation_log_7 = torch.as_tensor([[0] * dataset_len] * UNITS[2])
activation_log_9 = torch.as_tensor([[0] * dataset_len] * UNITS[3])
activation_log_11 = torch.as_tensor([[0] * dataset_len] * UNITS[4])

activation_logs = [
    activation_log_1,
    activation_log_4,
    activation_log_7,
    activation_log_9,
    activation_log_11
]

activation_log_names = [
    f"activation_log_{str(i)}" for i in LAYERS
]

##############################################################################
# defining forward hook
##############################################################################
def activation(module, input, output,
    measure=ACTIVATION_MEASUREMENT,
    ):
    """logs the activation of the units of a specific layer to the specified
    logging dictionary
    - keys: filenames
    - values: activation values, defined using a measurement function passed
    as a keyword argument
    note: the reference to output[0] assumes the batch submitted to the model
    only contains one image at a time 
    note: the reference to file_index is abusive and should be avoided in
    production code
    """
    units = range(output[0].size()[0])
    log_name = next(iter(module.state_dict()))
    log = module.state_dict()[log_name]
    for unit in units:
        #activation = float(measure(output[0][unit]))
        activation = measure(output[0][unit])
        log[unit, file_index] = activation

##############################################################################

# timestamp: exec clock starts
start = datetime.datetime.now()

# step 1
print("\n--- Loading network...")
model = torch.hub.load(
    'pytorch/vision:v0.10.0',
    'alexnet',
    pretrained=True
)

# step 2: forward hooks and forward buffers registrations
# the forward hook is the function that allows us to log the activations
# into the forward buffer; see def activation above

for i in range(len(LAYERS)):
    log_name = activation_log_names[i]
    model.features[LAYERS[i]].register_buffer(log_name, activation_logs[i])
    model.features[LAYERS[i]].register_forward_hook(activation)

# step 3: passing images to the network
print(f"\n--- Forward pass on images from {PATH} ({dataset_len} files)...")

checkpoints = [(i * int(dataset_len / 10)) for i in range(1, 11)]

for file_index in range(dataset_len):
    filename = f"{PATH}\\{file_list[file_index]}"
    helpers.network_forward_pass(model, filename)
    
    # progress indicator (for long file lists)
    if file_index in checkpoints:
        checkpoint = datetime.datetime.now()    
        ckpt_timestamp = checkpoint.strftime("%H:%M:%S")
        status = f"\tStill working (file {file_index} out of {dataset_len}"
        status += f", time: {ckpt_timestamp})"
        print(status)

# step 4
print("\n--- Aggregating layer logs into top 10 log...")
top10_log = {}

for layer in range(len(LAYERS)):
    layer_id = LAYERS[layer]
    for unit in range(UNITS[layer]):
        log = {}
        
        for file_id in range(dataset_len):
            log.update({file_id: activation_logs[layer][unit][file_id]})

        inverse_log = {value: key for key, value in log.items()}

        activation_values = [log[item] for item in log]
        activation_values.sort(reverse=True)

        top10 = []

        for rank in range(10):
            activ_value = activation_values[rank]
            activ_file_index = inverse_log[activ_value]
            top10 += [file_list[activ_file_index]]
            
        top10_log.update({(layer_id, unit): top10})

# step 5
print("\n--- Saving log...")
delimiter = "\n##############################################################"
date = datetime.datetime.now().strftime("%m%d")
log_filename = f"{MEASURE_STRING}_activation_log.txt"
log_file = open(log_filename, "w")
file_content = f"{MEASURE_STRING} activation for dataset {DATASET}"
file_content += f" logged on {date}" + delimiter

for layer in range(len(LAYERS)):
    layer_id = LAYERS[layer]
    for unit in range(UNITS[layer]):
        file_content += delimiter
        file_content += f"\n{layer_id}, {unit}:\n{top10_log[(layer_id, unit)]}"

log_file.write(file_content)
log_file.close()

# timestamp: exec clock stops
stop = datetime.datetime.now()    
timestamp = stop.strftime("%m%d_%H%M%S")

execution_time = str((stop - start).total_seconds())
print(f"\nExecuted in {execution_time} seconds")
