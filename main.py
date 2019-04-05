

import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD
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
    target_size=(299, 299),
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
model = InceptionV3(include_top=True, weights=None,classes=10)
# basemodel = VGG16(include_top=True, weights=None,classes=10)

model.compile(optimizer=SGD(lr=0.01, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
train_log = model.fit_generator(generator=train_generator,
                    steps_per_epoch=547,
                    epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=48,
                    callbacks=[history])

# #绘制acc-loss曲线
# history.loss_plot('epoch')
# # plot the training loss and accuracy
# plt.style.use("ggplot")
# plt.figure()
# plt.plot(np.arange(0, epochs), train_log.history["loss"], label="train_loss")
# # plt.plot(np.arange(0, epochs), train_log.history["val_loss"], label="val_loss")
# plt.plot(np.arange(0, epochs), train_log.history["acc"], label="train_acc")
# # plt.plot(np.arange(0, epochs), train_log.history["val_acc"], label="val_acc")
# plt.title("Training Loss and Accuracy on sar classifier")
# plt.xlabel("Epoch #")
# plt.ylabel("Loss/Accuracy")
# plt.legend(loc="upper right")
# plt.savefig("Loss_Accuracy_alexnet_{:d}e.jpg".format(epochs))
# plt.show()
