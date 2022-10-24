from PIL import Image
import helpers
import sys

LAYER = sys.argv[1]
UNIT = sys.argv[2]
target = f"l{LAYER}u{UNIT}"

log = "./logs/streetlab_max_activation_log_0427.txt"
save_path = f"./top10/top10_{target}.jpg"
square_side = 224
new_size = (square_side, square_side)

file_list, dataset_path = helpers.top_10(LAYER, UNIT, log)

output_image = Image.new("RGB", (len(file_list) * square_side, square_side))
imgx, imgy = output_image.size

for position in range(len(file_list)):
    path = f"{dataset_path}/{file_list[position]}"
    input_image = Image.open(path)
    buffer = input_image.resize(new_size)
    output_image.paste(buffer, (position * square_side, 0))

output_image.save(save_path, "jpeg")

dcm_save_path = f"./discrepancy_maps/{target}/top10_{target}.jpg"
dcm_size = (len(file_list) * square_side, 2 * square_side)
dcm_image = Image.new("RGB", dcm_size)
d_imgx, d_imgy = dcm_image.size

for position in range(len(file_list)):
    file = file_list[position]
    path = f"{dataset_path}/{file}"
    input_image = Image.open(path)
    buffer = input_image.resize(new_size)
    dcm_image.paste(buffer, (position * square_side, 0))
    file_number = file.rstrip(".jpg")
    dcm_path = f"./discrepancy_maps/{target}/{file_number}_dc_{target}.jpg"
    dcm = Image.open(dcm_path)
    dcm_image.paste(dcm, (position * square_side, square_side))
    
dcm_image.save(dcm_save_path, "jpeg")
