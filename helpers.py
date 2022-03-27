import torch
from PIL import Image
from torchvision import transforms


def tensor_from_file(filename):
    """preps the image stored in the image file and preprocesses it before
    classification by the model
    """
    image = Image.open(filename)

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


def network_forward_pass(model, filename):
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
            coordinates = (0, y_size[position])
        else:
            #coordinates = (position * max_x, 0)
            coordinates = (x_sizes[position], 0)

        output_image.paste(images[position], coordinates)

    output_image.save(filepath, "jpeg")
