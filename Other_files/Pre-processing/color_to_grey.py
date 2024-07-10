import os
import cv2

input_folder = '/Users/varshit_madi/Downloads/Image_PreProcessing/color_img'

output_folder = '/Users/varshit_madi/Downloads/Image_PreProcessing/grey_img'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

file_list = os.listdir(input_folder)

for file_name in file_list:
    if file_name.endswith(('.jpg', '.png', '.jpeg')):
        input_path = os.path.join(input_folder, file_name)

        color_image = cv2.imread(input_path)

        grayscale_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        output_path = os.path.join(output_folder, file_name)

        cv2.imwrite(output_path, grayscale_image)

print("Conversion complete. Grayscale images saved in:", output_folder)
