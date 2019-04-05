import pandas as pd
from tensorflow.python.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import load_img,img_to_array
import numpy as np
import h5py

h5file = h5py.File('singlesofa_data.h5', 'w')

data = pd.read_csv('mydata.csv',sep='$',encoding='utf-8')

anchorDir = data['anchor']
positiveDir = data['positive']
negativeDir = data['negative']
anchor_data = h5file.create_dataset('anchor',(len(anchorDir),224, 224 ,3))
positive_data = h5file.create_dataset('positive',(len(anchorDir),224, 224 ,3))
negative_data = h5file.create_dataset('negative',(len(anchorDir),224, 224 ,3))
batch_size = 32

rootDir = r'G:\毕设资料\dataset'

def convert_to_HDF5(dir, data):
    for i in np.arange(0, len(dir)):
        image = load_img(rootDir + dir[i], target_size=(224, 224))
        image = img_to_array(image)
        image = image/255
        # image = np.expand_dims(image, axis=0)
        # image = imagenet_utils.preprocess_input(image,mode='tf')
        print(i)
        data[i] = image

convert_to_HDF5(anchorDir, anchor_data)
convert_to_HDF5(positiveDir, positive_data)
convert_to_HDF5(negativeDir, negative_data)

h5file.close()

# x2 = image.img_to_array(img)
# x2 = np.expand_dims(x2, axis=0)
#
# x3 = image.img_to_array(img)
# x3 = np.expand_dims(x3, axis=0)


