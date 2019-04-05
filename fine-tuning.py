import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D

from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD,Adam
from model.graph import LossHistory


train_datagen = keras.preprocessing.image.ImageDataGenerator(
                                                             rotation_range=20,
                                                             width_shift_range=0.2,
                                                             height_shift_range=0.2,
                                                             horizontal_flip=True,
                                                             validation_split=0.08,
                                                             rescale=1 / 255.0)



datasetdir = 'G:\My Projects\LearnPython\slim\images'
# 1.从图像目录中加载数据
train_generator = train_datagen.flow_from_directory(
    datasetdir,
    target_size=(224, 224),
    batch_size=32,
    subset='training',
    class_mode='categorical')

validation_generator = train_datagen.flow_from_directory(
    datasetdir,
    target_size=(224, 224),
    batch_size=32,
    subset='validation',
    class_mode='categorical')


epochs = 21

history = LossHistory()


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
  x = GlobalAveragePooling2D(name='avg_pool')(x)
  x = Dropout(rate=0.5)(x)#增加的dropout层不知道有没有用
  # x = Dense(FC_SIZE, activation='relu')(x) #new FC layer, random init
  predictions = Dense(nb_classes, activation='softmax',kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None))(x) #new softmax layer
  model = Model(inputs=base_model.input, outputs=predictions)
  return model



# 定义网络框架
base_model = InceptionV3(weights='G:\My Projects\LearnPython\model-search\model\weights\inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False) # 预先要下载no_top模型
model = add_new_last_layer(base_model, 10)              # 从基本no_top模型上添加新层
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

train_log = model.fit_generator(generator=train_generator,
                    steps_per_epoch=547,
                    epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=48,
                    callbacks=[history])


