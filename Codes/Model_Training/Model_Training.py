'''
This program imports the created model and dataset and trains them
'''

from pathlib import Path
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from scipy.io import loadmat


current_dir = Path(__file__).parent
model_path = current_dir / '../../Models/Model_1.01.h5'
model_path = model_path.resolve()

mat_path = current_dir / '../../Datasets/Harvard_University/Harvard_University_Images.mat'
mat_path = mat_path.resolve()

num_epochs = 50
batch_size = 1

def data_generator(indices, batch_size, images_bw, images_rgb):
    while True:
        np.random.shuffle(indices)  
        for start_idx in range(0, len(indices), batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            yield images_bw[batch_indices], images_rgb[batch_indices]
            
def BW_Resize(images):
    img = np.repeat(images, 3, axis=-1)
    return img


mat_file = loadmat(mat_path)
Img_RGB = mat_file['RGB']
Img_BW = BW_Resize(mat_file['BW'])

model = load_model(model_path)
model.compile(optimizer=Adam(), loss='mean_squared_error')

num,_,_,_ = Img_RGB.shape
indices = np.arange(num)

train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)

train_generator = data_generator(train_indices, batch_size, Img_BW/255.0, Img_RGB/255.0)
test_generator = data_generator(test_indices, batch_size, Img_BW/255.0, Img_RGB/255.0)

model.fit(train_generator,
          steps_per_epoch=len(train_indices) // batch_size,
          epochs=num_epochs,
          validation_data=test_generator,
          validation_steps=len(test_indices) // batch_size)

model.save(model_path)
