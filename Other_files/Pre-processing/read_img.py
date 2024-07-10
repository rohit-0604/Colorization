import os
import numpy as np
from PIL import Image

image_path = '/Users/varshit_madi/Downloads/Image_PreProcessing/color_img'

file_list = os.listdir(image_path)

for image_file in file_list:

    full_path = os.path.join(image_path, image_file)
    img = Image.open(full_path)

    print(
        f"Image: {image_file}, Image format: {img.format}, Size: {img.size}, Mode: {img.mode}")

    n = np.asarray(img)
    print(n)
    print()
