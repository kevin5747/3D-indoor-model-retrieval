import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D, Input

from tensorflow.keras import backend as K


def triplet_loss(inputs):
    anchor, positive, negative = inputs
    dis_pos = K.sum(K.square(anchor - positive), axis=1, keepdims=True)
    dis_neg = K.sum(K.square(anchor - negative), axis=1, keepdims=True)
    dis_pos = K.sqrt(dis_pos)
    dis_neg = K.sqrt(dis_neg)
    a1 = 17
    d1 = K.maximum(0.0, dis_pos - dis_neg + a1)
    return K.mean(d1)


def create_model():
    anchor_input = Input(shape=(224, 224, 3), dtype='float32', name='anchor_input')
    positive_input = Input(shape=(224, 224, 3), dtype='float32', name='positive_input')
    negative_input = Input(shape=(224, 224, 3), dtype='float32', name='negative_input')

    FEATURE_NUM = 1024
    base_model = InceptionV3(weights=r'D:\python\model-search\model\weights\inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
                             include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    # x = Dropout(rate=0.6)(x)#增加的dropout层不知道有没有用
    # x = Dense(FC_SIZE, activation='relu')(x) #new FC layer, random init
    predictions = Dense(FEATURE_NUM, activation='relu',
                        kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None))(
        x)  # new softmax layer
    feature_model = Model(inputs=base_model.input, outputs=predictions)

    anchor_feature = feature_model(anchor_input)
    positive_feature = feature_model(positive_input)
    negative_feature = feature_model(negative_input)

    inputs = [anchor_input, positive_input, negative_input]
    outputs = [anchor_feature, positive_feature, negative_feature]

    model = Model(inputs=inputs, outputs=outputs)
    model.add_loss(triplet_loss(outputs))
    return  model