import os

image_folder_path = '/Users/varshit_madi/Downloads/Image_PreProcessing/color_img'

file_list = os.listdir(image_folder_path)

image_files = [file for file in file_list if file.lower().endswith(
    ('.jpg', '.png', '.jpeg'))]

for index, image_file in enumerate(image_files):
    original_path = os.path.join(image_folder_path, image_file)
    new_name = f'image_{index + 1}.jpg'
    new_path = os.path.join(image_folder_path, new_name)

    os.rename(original_path, new_path)
