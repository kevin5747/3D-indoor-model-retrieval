import numpy as np
import tensorflow.keras as keras
from model.create_model import create_model, triplet_loss
from tensorflow.python.keras.optimizers import SGD,Adam
from model.graph import LossHistory
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from scipy.interpolate import spline, interp1d
import h5py

def plot_loss(lossList):
    # xnew = np.linspace(0, len(lossList), len(lossList)*0.5)
    # loss_smooth = spline(np.arange(len(lossList)), lossList, xnew)#损失函数平滑处理
    plt.style.use("ggplot")
    plt.figure()
    # loss
    # plt.plot(xnew, loss_smooth, 'g', label='loss_smo0th')
    plt.plot(range(len(lossList)), lossList, 'r', label='train loss')#原版曲线
    plt.grid(True)
    plt.xlabel('batch')
    plt.ylabel('loss')
    plt.legend(loc="upper right")
    plt.show()



model = create_model()
model.load_weights('my_model_weights.h5')
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9))



HDF5 = h5py.File('singlesofa_data.h5', 'r')
anchor_data = HDF5['anchor']
positive_data = HDF5['positive']
negative_data = HDF5['negative']
BATCH_SIZE = 32
lossList = []
EPOCHS = 100
TOTAL_STEPS = len(anchor_data)
STEPS_PER_EPOCH = int(TOTAL_STEPS/BATCH_SIZE)+1
# STEPS_PER_EPOCH = 100
for i in range(0,EPOCHS):
    if i != 0:
        plot_loss(lossList)
        model.save_weights('my_model_weights.h5')
    for j in range(0,STEPS_PER_EPOCH):
        start = j*BATCH_SIZE
        end = (j+1)*BATCH_SIZE
        if end > TOTAL_STEPS :#防止溢出
            if start < TOTAL_STEPS:
                end = TOTAL_STEPS
            else:
                continue
        total_input = {'anchor_input': anchor_data[start:end, :], 'positive_input': positive_data[start:end, :],
                       'negative_input': negative_data[start:end, :]}
        loss = model.train_on_batch(total_input)
        lossList.append(loss)#每批次记录loss
        if(loss != 0.0):
            print("epoch:%d steps:[%d/%d] loss:%f"%(i+1, j+1, STEPS_PER_EPOCH, loss))

