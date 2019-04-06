import pandas as pd
from model.create_model import create_model
from tensorflow.python.keras.preprocessing.image import load_img,img_to_array
from tensorflow.python.keras.optimizers import SGD
import numpy as np
import h5py

std_pic_list = pd.read_csv('../cvs/std_pic.csv',sep='$',encoding='utf-8')['std_pic']
h5file = h5py.File('std_feature.h5', 'w')
dt = h5py.special_dtype(vlen=str)
dir = h5file.create_dataset('std_dir',data=std_pic_list, dtype=dt)
feature = h5file.create_dataset('std_feature',(len(std_pic_list),1024))

ROOT_DIR = r'G:\毕设资料\dataset'
model = create_model()
model.load_weights('../../my_model_weights.h5')
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9))

def convert_to_HDF5(dir, data):
    for i in np.arange(0, len(dir)):
        image = load_img(ROOT_DIR + dir[i], target_size=(224, 224))
        image = img_to_array(image)
        image = image/255
        image = np.expand_dims(image, axis=0)
        print(i)
        feature, _, __ = model.predict_on_batch([image, image, image])
        data[i] = feature

convert_to_HDF5(std_pic_list,feature)
h5file.close()