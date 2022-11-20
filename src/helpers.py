import torch
from PIL import Image
import cv2 as cv
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


def network_forward_pass(model, filename, mode="hub-model", gpu=True):
    if mode == "avatar":
        input_tensor = tensor_from_avatar_file(filename)
    else:
        input_tensor = tensor_from_file(filename)

    # creates a mini-batch as expected by the model
    input_batch = input_tensor.unsqueeze(0)
    
    # moves the input and model to GPU for speed if available
    try: 
        if gpu and torch.cuda.is_available():
            with torch.no_grad():
                input_batch = input_batch.to('cuda')
                #model.to('cuda')
    except RuntimeError: # out of memory
        pass

    # passes the input batch to the model
    with torch.no_grad():
        output = model(input_batch)
        
    del input_batch

    return output


def stack(image_list, filepath, direction="row"):
    """image_list: Python list of image filepaths
    filepath: path to save to
    direction: direction to use in stacking
        defaults to "row"
        accepts "column"
    output: an image file
    """
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

    if direction == "column":
        output_size = (max_x, sum_y)
    else:
        output_size = (sum_x, max_y)

    output_image = Image.new("RGB", output_size)

    x_buffer = 0
    y_buffer = 0
    for position in range(len(images)):
        if direction == "column":
            coordinates = (0, y_buffer)
            y_buffer += y_sizes[position]
        else:
            coordinates = (x_buffer, 0)
            x_buffer += x_sizes[position]

        output_image.paste(images[position], coordinates)

    output_image.save(filepath, "png")


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
    array_sum = 0
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
    
    result.save(target_filename, "png")


def calibrate(filepath, target_filepath):
    source = Image.open(filepath)
    x_size = source.size[0]
    y_size = source.size[1]
    output = Image.new("RGB", (x_size, y_size))

    name = filepath.split("/")[-1]    
    tail = f"/{name}"
    path = filepath.rstrip(tail)
    
    tensor = np.asarray(source)
    # np.nonzero returns three arrays of indices :
    # row (y), column (x), depth (band), in this order
    try:
        min_y = min(np.nonzero(tensor)[0])
        max_y = max(np.nonzero(tensor)[0])
        min_x = min(np.nonzero(tensor)[1])
        max_x = max(np.nonzero(tensor)[1])
    except:
        output.save(target_filepath, "png")
        print(f"(blank) calibrated file saved in {target_filepath}")
        return

    largest_dimension = max(max_x - min_x, max_y - min_y)

    if largest_dimension == 0:
        output.save(target_filepath, "png")
        print(f"(blank) calibrated file saved in {target_filepath}")
        return

    size = (largest_dimension, largest_dimension)
    box = (min_x, min_y, max_x, max_y)

    patch = source.resize(size, box=box)
    
    x_corner = int(x_size / 2 - largest_dimension / 2)
    y_corner = int(y_size / 2 - largest_dimension / 2)
    corner = (x_corner, y_corner)
    
    output.paste(patch, box=corner)

    output.save(target_filepath, "png")
    #print(f"calibrated file saved in {target_filepath}")


def resize(filepath, save_filepath, new_size=(224, 224)):
    """resize target image and stores it in base folder for discrepancy
    map computation
    """

    input_image = Image.open(filepath)

    output_image = input_image.resize(new_size)

    filename = filepath.split("/")[-1]
    output_image.save(save_filepath, "jpeg")


def occlusion(filepath, occluded_dir, stride=3, occluder=11):
    """occluded_dir: to specify the directory where the occluded images must
    be saved
    """
    name = filepath.split("/")[-1].rstrip(".jpg").rstrip(".png")
    occluded_list = os.listdir(occluded_dir)

    if name in occluded_list:
        print("Occluded images already exist. Skipping...")
        return None
    else:
        # timestamp: exec clock starts
        start = datetime.datetime.now()

        output_dir = os.path.join(occluded_dir, name)
        os.mkdir(output_dir)    

        OCCLUDER_SIZE = occluder
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

        status_template = """processing image: {}
            image size: {}
            window size: {} pixels on x axis, {} on y axis
            stride: {}
            nb of steps in x direction: {}
            nb of steps in y direction: {}
            """

        status_update = status_template.format(
            filepath,
            size,
            x_window, y_window,
            STRIDE,
            x_steps, y_steps
        )

        print(status_update)

        #each pixel of each occluding window is given a noisy rgb value
        #computed using rng.normal (loc = mu (mean), scale = sigma (stdev))
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

            output_filepath = f"{output_dir}/{name}_{step[0]:03d}-{step[1]:03d}.png"
            output_image.save(output_filepath, "png")

        input_image.close()
        print("occluded images successfully created")

        # timestamp: exec clock stops
        stop = datetime.datetime.now()    
        execution_time = (stop - start).total_seconds()
        end_message =  f"Time: {execution_time:.2f} seconds\n"
        print(end_message)


def top_10(layer, unit, log):
    """warning: heavily relies on the structure of the log file"""
    log_file = open(log, 'r')
    file_as_list = [line for line in log_file]
    
    # target unit log retrieval: using the "L, U:" line as a checkpoint
    checkpoint = file_as_list.index(f"{layer}, {unit}:\n")
    target_line = file_as_list[checkpoint + 1]

    # building filepath list from log line:
    # ['relpath/to/images/image0.jpg', .., 'relpath/to/images/image9.jpg']
    buffer_list = target_line.rstrip("']\n").lstrip("['").split("', '")
    image_list = []

    for filename in buffer_list:
        filename = filename.replace("..", os.getcwd())
        image_list.append(filename)

    return image_list


def max_extractor(filepath, target_filepath):
    image = Image.open(filepath)
    output_image = Image.new("L", image.size)
    
    array = np.asarray(image.convert(mode="L"))
    center = np.unravel_index(np.argmax(array, axis=None), array.shape)
    center = (float(center[0]), float(center[1]))

    threshold = 25

    array = np.where(array < threshold, 0, array)

    contours, _ = cv.findContours(
        array,
        mode=cv.RETR_EXTERNAL,
        method=cv.CHAIN_APPROX_NONE
    )

    tests = np.asarray(
        [cv.pointPolygonTest(contour, center, True) for contour in contours]
    )
    try:
        max_act_contour = np.argmax(tests)
    except:
        "Error during contour extraction; saving blank image"
        output_image.save(target_filepath, "png")
        return

    # array of distances to main contour
    main_contour = [[cv.pointPolygonTest(contours[max_act_contour], (x, y), True) for x in range(image.size[0])] for y in range(image.size[1])]
    
    for x in range(image.size[0]):
        for y in range(image.size[1]):
            # if the current pixel is within the main contour, value, else 0
            if main_contour[y][x] > 0:
                value = image.getpixel((x, y))[0] #assumes tuple with three identical coordinates
                output_image.putpixel((x, y), value)

    output_image.save(target_filepath, "png")
