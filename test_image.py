from PIL import Image

#WHITE = (255, 255, 255)    # white
#RED = (255, 0, 0)        # red
#BLUE = (0, 0, 255)        # blue
#GREEN = (0, 255, 0)        # green
#MAGENTA = (255, 0, 255)      # magenta
#YELLOW = (255, 255, 0)      # yellow
#CYAN = (0, 255, 255)      # cyan
#OCEAN = (0, 127, 255)      # ocean blue
BLACK = (0, 0, 0)          # black

#color = MAGENTA

#background_color = (255, 255, 255)
background_color = BLACK

#thickness = 3 # thin; grid
#thickness = 5 # default
#thickness = 10 # thick
#thickness = 100 # wide

x_size = 256
y_size = x_size

filename = f"black"

output_image = Image.new("RGB", (x_size, y_size))

#"""
# background
for x in range(x_size):
    for y in range(y_size):
        output_image.putpixel((x, y), background_color)
#"""

"""
# vertical lines
for i in range(1, 9):
    for x in range (30 * i, 30 * i + thickness):
        for y in range(y_size):
            output_image.putpixel((x, y), color)
"""

"""
# horizontal lines
for i in range(1, 9):
    for x in range (x_size):
        for y in range(30 * i, 30 * i + thickness):
            output_image.putpixel((x, y), color)
"""

folder = "discrepancy_maps"
filepath = f"{folder}/{filename}.jpg"

output_image.save(filepath, "jpeg")
