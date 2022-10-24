##############################################################################
import sys
from PIL import Image

NEW_SIZE = (224, 224)

filename = sys.argv[1]

input_image = Image.open(filename)

output_image = input_image.resize(NEW_SIZE)

output_filename = f"resized_{filename}"
output_image.save(output_filename, "jpeg")
