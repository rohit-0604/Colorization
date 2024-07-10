import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Conv2D, UpSampling2D
from tensorflow.python.keras.models import Sequential
from keras.utils import img_to_array, load_img, array_to_img
from sklearn.model_selection import train_test_split
import os
from matplotlib import pyplot as plt

# Replace with your actual directories
color_images_dir = r'C:\Users\Varsha\Desktop\Ae\Image_PreProcessing\color_img_png'
grayscale_images_dir = r'C:\Users\Varsha\Desktop\Ae\Image_PreProcessing\grey_img_png'

# Load images into memory
def load_images(directory,target_size=(256,256),color_mode='grayscale'):
    images = []
    for filename in os.listdir(directory):
        image_path=os.path.join(directory,filename)
        image = img_to_array(load_img(image_path,color_mode=color_mode,target_size=target_size))
        #image=tf.image.resize(image,[256,256])
        images.append(image)
    return np.array(images)

print("Loading Images")
color_images = load_images(color_images_dir,target_size=(256,256),color_mode='rgb')
grayscale_images = load_images(grayscale_images_dir,target_size=(256,256),color_mode='grayscale')
print("Images loaded")

# Normalize pixel values to be between 0 and 1
color_images = color_images.astype('float32') / 255.
grayscale_images = grayscale_images.astype('float32') / 255.

# Split the data
X_train, X_test, y_train, y_test = train_test_split(grayscale_images, color_images, test_size=0.1)

# Define the model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(256, 256, 1)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
#model.add(UpSampling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
#model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(3, (3, 3), activation='sigmoid', padding='same'))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

#X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
#X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

# Train the model
model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test))

new_color_image_path=r'C:\Users\Varsha\Desktop\Ae\Image_PreProcessing\color_img_png\image_70.png'
new_color_image=img_to_array(load_img(new_color_image_path,color_mode='rgb',target_size=(256,256)))

new_bw_img=img_to_array(load_img(new_color_image_path,color_mode='grayscale',target_size=(256,256)))

new_bw_img=new_bw_img.astype('float32')/255

new_bw_img=new_bw_img.reshape(1,256,256,1)

# Predict on a sample grayscale image
output = model.predict(new_bw_img)

# Show the grayscale input image, true image, and the prediction from the model
plt.figure(figsize=(10, 3))
plt.subplot(1, 3, 1)
plt.title('Input')
plt.imshow(new_bw_img.squeeze(),cmap='gray')
plt.subplot(1, 3, 2)
plt.title('True')
plt.imshow(array_to_img(new_color_image))
plt.subplot(1, 3, 3)
plt.title('Predicted')
plt.imshow(array_to_img(output[0]))
plt.show()