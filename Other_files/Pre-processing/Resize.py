from PIL import Image
import os

# Set the path to your image folder
image_folder_path = "/Users/varshit_madi/Downloads/Image_PreProcessing/grey_img"

# Set the target size for resizing
target_size = (1280, 720)

# Create a new folder to store resized images
output_folder_path = '/Users/varshit_madi/Downloads/Image_PreProcessing/resize_grey_img'
os.makedirs(output_folder_path, exist_ok=True)

# Iterate through each image in the folder
for filename in os.listdir(image_folder_path):
    image_path = os.path.join(image_folder_path, filename)

    # Open the image using PIL
    img = Image.open(image_path)

    # Resize the image
    resized_img = img.resize(target_size)

    # Save the resized image to the output folder
    output_path = os.path.join(output_folder_path, filename)
    resized_img.save(output_path)

print("Resizing complete. Resized images saved in:", output_folder_path)
