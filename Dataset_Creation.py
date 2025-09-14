'''
This Program generates a .mat file that has GreyScale and Color Images from images that contain Color images
'''


import os
import numpy as np
from PIL import Image
from scipy.io import savemat
from tensorflow.keras.preprocessing import image
from pathlib import Path


current_dir = Path(__file__).parent

#your matfile name, here I used 'Harvard_University_Images.mat'
mat_path = current_dir / '../../Datasets/Harvard_University/Harvard_University_Images.mat'
mat_path = mat_path.resolve()

#folder_path = <path to your images>

#specify the shape of the images
target_size = (250,250)

def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path)
            img = image.smart_resize(img, target_size)
            images.append(img)
    return images

def Color2BW(images):
    img = np.mean(images, axis=-1, keepdims=True) 
    img = np.repeat(img, 1, axis=-1)
    return img

images = load_images_from_folder(folder_path)

Img_RGB = np.array(images, dtype = np.uint8)
Img_BW = np.array(Color2BW(Img_RGB), dtype = np.uint8)

Data = {'BW': Img_BW, 'RGB': Img_RGB}

savemat(mat_path,Data)