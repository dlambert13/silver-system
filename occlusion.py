##############################################################################
import sys
from PIL import Image
import numpy.random as random

NAME = sys.argv[1]
OCCLUDER_SIZE = 11
STRIDE = 3

rng = random.default_rng()

filename = f"discrepancy_maps/{NAME}.jpg"
input_image = Image.open(filename)

size = input_image.size

x_window = OCCLUDER_SIZE
y_window = OCCLUDER_SIZE

#steps: coordinates of the pixel where each occluding window originates
#occluder will cover x_window pixels from x_step to x_step + x_window - 1
x_steps = 1 + int((size[0] - x_window) / 3)
y_steps = 1 + int((size[1] - y_window) / 3)

steps = [
	(x_step, y_step)
	for x_step in range(x_steps)
	for y_step in range(y_steps)
]

print(
	f"processing image: {filename}\n"
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

	output_filepath = f"occluded/{NAME}/{NAME}_{step[0]:03d}-{step[1]:03d}.jpg"

	output_image.save(output_filepath, "jpeg")

input_image.close()
print("occluded images successfully created")