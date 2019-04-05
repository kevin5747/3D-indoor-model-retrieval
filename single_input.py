import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D, Input

from tensorflow.keras.layers import Conv2D, MaxPooling2D, concatenate
from tensorflow.keras.optimizers import SGD,Adam
from model.graph import LossHistory
from tensorflow.keras import backend as K
import h5py
BATCH_SIZE = 3

def triplet_loss(inputs):
    inputs = K.l2_normalize(inputs, axis=1)
    anchor = inputs[0:BATCH_SIZE,:]
    positive = inputs[BATCH_SIZE:2*BATCH_SIZE,:]
    negative = inputs[2*BATCH_SIZE:,:]
    dis_pos = K.sum(K.square(anchor - positive), axis=1, keepdims=True)
    dis_neg = K.sum(K.square(anchor - negative), axis=1, keepdims=True)
    # dis_pos = K.sqrt(dis_pos)
    # dis_neg = K.sqrt(dis_neg)
    a1 = 8
    d1 = K.maximum(0.0, dis_pos - dis_neg + a1)
    return K.mean(d1)






# 添加新层
def add_new_last_layer(base_model, nb_classes):
  """
  添加最后的层
  输入
  base_model和分类数量
  输出
  新的keras的model
  """
  x = base_model.output
  feature = Flatten(name='feature')(x)
  predictions = Dropout(0.5)(feature)
  # x = Dropout(rate=0.5)(x)#增加的dropout层不知道有没有用
  # x = Dense(FC_SIZE, activation='relu')(x) #new FC layer, random init
  # predictions = Dense(nb_classes, activation='relu',kernel_initializer=keras.initializers.TruncatedNormal(mean=0.1, stddev=0.05, seed=None))(x) #new softmax layer
  model = Model(inputs=base_model.input, outputs=predictions)
  return model

# 定义网络框架
base_model = InceptionV3(weights='model/weights/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False) # 预先要下载no_top模型
model = add_new_last_layer(base_model, 32)              # 从基本no_top模型上添加新层
model.add_loss(triplet_loss(model.output))
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9))

HDF5 = h5py.File('singlesofa_data.h5', 'r')
anchor_data = HDF5['anchor']
positive_data = HDF5['positive']
negative_data = HDF5['negative']

EPOCHS = 1
STEPS_PER_EPOCH = int(len(anchor_data)/BATCH_SIZE)
for i in range(0,EPOCHS):
    for j in range(0,STEPS_PER_EPOCH):
        print(j)
        new_data = np.vstack((anchor_data[j*BATCH_SIZE:(j+1)*BATCH_SIZE, :],positive_data[j*BATCH_SIZE:(j+1)*BATCH_SIZE, :],negative_data[j*BATCH_SIZE:(j+1)*BATCH_SIZE, :]))

        print(model.train_on_batch(new_data))
# j = 0
# new_data = np.vstack((anchor_data[j*BATCH_SIZE:(j+1)*BATCH_SIZE, :],positive_data[j*BATCH_SIZE:(j+1)*BATCH_SIZE, :],negative_data[j*BATCH_SIZE:(j+1)*BATCH_SIZE, :]))
# dis = model.predict(new_data)
# print(dis)
# anchor = dis[0:BATCH_SIZE, :]
# positive = dis[BATCH_SIZE:2 * BATCH_SIZE, :]
# negative = dis[2 * BATCH_SIZE:, :]
# print(anchor)
# print(negative)
# dis_pos = np.sum(np.square(anchor - positive), axis=1, keepdims=True)
# dis_neg = np.sum(np.square(anchor - negative), axis=1, keepdims=True)
# #
# print(dis_pos)
# print(dis_neg)
# a1 = 17
#
# d1 = np.maximum(0.0, dis_pos - dis_neg + a1)
# print(np.mean(d1))
# keras.utils.plot_model(model, to_file='model1.png',show_shapes=True)

# HDF5 = h5py.File('singlesofa_data.h5', 'r')
# anchor_data = HDF5['anchor']
# positive_data = HDF5['positive']
# negative_data = HDF5['negative']
# BATCH_SIZE = 1
# total_input = {'anchor_input':anchor_data[0:BATCH_SIZE], 'positive_input':positive_data[0:BATCH_SIZE], 'negative_input':negative_data[0:BATCH_SIZE]}


# model.fit(total_input,batch_size=1,epochs=20)

# train_datagen = keras.preprocessing.image.ImageDataGenerator()
#
# model.fit_generator(train_datagen.flow([anchor_data,positive_data,negative_data], batch_size=1),
#                     steps_per_epoch=len(anchor_data) / 1, epochs=1)

# for i in range(0,5):
#     model.train_on_batch(total_input)