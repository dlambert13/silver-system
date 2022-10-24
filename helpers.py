import torch
from PIL import Image, ImageOps
from torchvision import transforms
import numpy as np
import datetime
import os


def predict(path, model, preprocess):
    input_image = Image.open(path)
    if input_image.size[0] != 3:
        input_image = input_image.convert('RGB')
        input_tensor = preprocess(input_image)
    else:
        input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    output = model(input_batch)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
    label = round(probabilities.detach().numpy()[0], 0)
    confidence = probabilities.detach().numpy()[0]

    return label, confidence


def tensor_from_file(filename):
    """preps the image stored in the image file and preprocesses it before
    classification by the model
    """
    image = Image.open(filename)

    #to handle single-band images (TODO: PIL: ImageMode.getMode ?)
    if image.getbands() != ('R', 'G', 'B'):
        image = Image.merge("RGB", [image, image, image])

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    tensor = preprocess(image)

    image.close()

    return tensor


def tensor_from_avatar_file(filename):
    """preps the image stored in the image file and preprocesses it before
    classification by the model
    specific version for the streetlab-specific model
    TODO:
    -rewrite original activation script for backward compatibility
    -remove this duplicate function
    """
    image = Image.open(filename)

    #to handle single-band images (TODO: PIL: ImageMode.getMode ?)
    if image.getbands() != ('R', 'G', 'B'):
        image = Image.merge("RGB", [image, image, image])

    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])

    tensor = preprocess(image)

    image.close()

    return tensor


def network_forward_pass(model, filename, mode="hub-model"):
    if mode == "avatar":
        input_tensor = tensor_from_avatar_file(filename)
    else:
        input_tensor = tensor_from_file(filename)

    # creates a mini-batch as expected by the model
    input_batch = input_tensor.unsqueeze(0)
    
    # moves the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    # passes the input batch to the model
    with torch.no_grad():
        output = model(input_batch)

    return output


def stack(image_list, filepath, direction="row"):
    images = []

    for name in image_list:
        image = Image.open(name)
        images += [image]

    x_sizes = [image.size[0] for image in images]
    y_sizes = [image.size[1] for image in images]

    max_x = max(x_sizes)
    max_y = max(y_sizes)
    
    sum_x = 0
    sum_y = 0
    for i in range(len(images)):
        sum_x += x_sizes[i]
        sum_y += y_sizes[i]

    #to induce shift in size/position for loop:
    x_sizes = [0] + x_sizes
    y_sizes = [0] + y_sizes

    if direction == "column":
        #output_size = (max_x, len(images) * max_y)
        output_size = (max_x, sum_y)
    else:
        #output_size = (len(images) * max_x, max_y)
        output_size = (sum_x, max_y)

    output_image = Image.new("RGB", output_size)

    for position in range(len(images)):
        if direction == "column":
            #coordinates = (0, position * max_x)
            coordinates = (0, y_sizes[position])
        else:
            #coordinates = (position * max_x, 0)
            coordinates = (x_sizes[position], 0)

        output_image.paste(images[position], coordinates)

    output_image.save(filepath, "jpeg")


def mask_overlay(background, mask, savepath):
    back_im = Image.open(background)
    mask_im = Image.open(mask)
    mask_im = mask_im.convert("L")
    factor = 5
    
    x_size = back_im.size[0]
    y_size = back_im.size[1]

    back_pixels = back_im.load()
    mask_pixels = mask_im.load()

    for x in range(x_size):
        for y in range(y_size):
            mask_value = mask_pixels[x, y]
            if mask_value != 0:
                back_value = back_pixels[x, y]
                back_r = back_value[0]
                back_g = back_value[0]
                back_b = back_value[0]
                updated_value = (
                    back_r + factor * mask_value,
                    back_g + factor * mask_value,
                    back_b + factor * mask_value)    
                back_im.putpixel((x, y), updated_value)

    back_im.save(savepath, "png")


def add(file_list, target_filename):    
    for file in file_list:
        image = Image.open(file)
        np_array = np.asarray(image)
        if file_list.index(file) == 0:
            image_shape = np_array.shape
            array_sum = np.zeros(image_shape)
        array_sum += np_array

    array_max = np.max(array_sum)
    array_min = np.min(array_sum)
    normalized_array = array_sum - array_min
    normalized_array *= 255 / (array_max - array_min)
    
    result_array = normalized_array.astype(dtype="uint8")
    result = Image.fromarray(result_array)
    result.save(target_filename, "jpeg")


def mean(file_list, target_filename):
    length = len(file_list)
    tensor_sum = np.zeros((224, 224, 3), dtype="uint8")
    
    for file in file_list:
        image = Image.open(file)
        tensor = np.asarray(image)
        tensor_sum += tensor

    tensor_mean = np.array(tensor_sum / length, dtype="uint8")

    result = Image.fromarray(tensor_mean)
    result.save(target_filename, "jpeg")


def calibrate(filepath, target_filepath):
    #try threshold:     
    #try opencv:
    source = Image.open(filepath)

    name = filepath.split("/")[-1]    
    tail = f"/{name}"
    path = filepath.rstrip(tail)
    
    tensor = np.asarray(source)
    """
    values_list = np.unique(tensor)
    
    tensor_nonz_min = values_list[1]
    tensor_max = values_list[-1]
    THRESHOLD = (tensor_max - tensor_nonz_min) / 5
    """
    THRESHOLD = 10
    tensor = np.where(tensor > THRESHOLD, tensor, 0)

    #center = np.unravel_index(np.argmax(tensor, axis=None), tensor.shape)
    # np.nonzero returns three arrays of indices :
    # row (y), column (x), depth (band), in this order
    min_y = min(np.nonzero(tensor)[0])
    max_y = max(np.nonzero(tensor)[0])
    min_x = min(np.nonzero(tensor)[1])
    max_x = max(np.nonzero(tensor)[1])

    largest_dimension = max(max_x - min_x, max_y - min_y)

    size = (largest_dimension, largest_dimension)
    box = (min_x, min_y, max_x, max_y)

    patch = source.resize(size, box=box)

    x_size = source.size[0]
    y_size = source.size[1]
    output = Image.new("RGB", (x_size, y_size))
    
    x_corner = int(x_size / 2 - largest_dimension / 2)
    y_corner = int(y_size / 2 - largest_dimension / 2)
    corner = (x_corner, y_corner)
    
    output.paste(patch, box=corner)

    output.save(target_filepath, "jpeg")
    print(f"calibrated file saved in {target_filepath}")


def resize(filepath):
    """resize target image and stores it in base folder for discrepancy
    map computation
    """
    NEW_SIZE = (224, 224)

    input_image = Image.open(filepath)

    output_image = input_image.resize(NEW_SIZE)

    filename = filepath.split("/")[-1]
    output_filepath = f"discrepancy_maps/0_base/resized_{filename}"
    output_image.save(output_filepath, "jpeg")


def occlusion(filepath, stride=3):
    name = filepath.split("/")[-1].rstrip(".jpg").rstrip(".png")
    occluded_list = os.listdir("./occluded")

    if name in occluded_list:
        print("Occluded images already exist. Exiting...")
        return None
    else:
        # timestamp: exec clock starts
        start = datetime.datetime.now()

        output_dir = f"occluded/{name}"
        os.mkdir(output_dir)    

        OCCLUDER_SIZE = stride
        STRIDE = stride
        
        rng = np.random.default_rng()

        input_image = Image.open(filepath)

        size = input_image.size

        x_window = OCCLUDER_SIZE
        y_window = OCCLUDER_SIZE

        #steps: coordinates of the pixel where each occluding window originates
        #occluder will cover x_window pixels from x_step to x_step + x_window - 1
        x_steps = 1 + int((size[0] - x_window) / STRIDE)
        y_steps = 1 + int((size[1] - y_window) / STRIDE)

        steps = [
            (x_step, y_step)
            for x_step in range(x_steps)
            for y_step in range(y_steps)
        ]

        print(
            f"processing image: {filepath}\n"
            + f"image size: {size}\n"
            + f"window size: {x_window} pixels on x axis, {y_window} on y axis\n"
            + f"stride: {STRIDE}\n"
            + f"nb of steps in x direction: {x_steps}\n"
            + f"nb of steps in y direction: {y_steps}\n"
        )

        #each pixel of each occluding window is given a noisy rgb value
        #computed using rng.normal (loc = mu, scale = sigma)
        for step in steps:
            x_step = step[0] * STRIDE
            y_step = step[1] * STRIDE

            output_image = input_image.copy()
            putpixel = output_image.putpixel

            for x in range(x_step, x_step + x_window):
                for y in range(y_step, y_step + y_window):
                    r_noise = int(rng.normal(loc=120, scale=60))
                    g_noise = int(rng.normal(loc=120, scale=60))
                    b_noise = int(rng.normal(loc=120, scale=60))

                    putpixel((x, y), (r_noise, g_noise, b_noise))

            output_filepath = f"{output_dir}/{name}_{step[0]:03d}-{step[1]:03d}.jpg"
            output_image.save(output_filepath, "jpeg")

        input_image.close()
        print("occluded images successfully created")

        # timestamp: exec clock stops
        stop = datetime.datetime.now()    
        execution_time = str((stop - start).total_seconds())
        end_message =  f"\nTime: {execution_time} seconds\n"
        print(end_message)


def top_10(layer, unit, log):
    log_file = open(log, 'r')
    file_as_list = [line for line in log_file]
    
    # target unit log retrieval
    checkpoint = file_as_list.index(f"{layer}, {unit}:\n")
    target_line = file_as_list[checkpoint + 1]

    buffer_list = target_line.rstrip("']\n").lstrip("['").split("', '")
    image_list = []

    for filename in buffer_list:
        if filename.find("\\") != -1:
            filename = filename.replace("\\\\", "/")
        image_list.append(filename)

    return image_list
