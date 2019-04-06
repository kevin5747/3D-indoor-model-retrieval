from tensorflow.python.keras.optimizers import SGD
from model.create_model import create_model
from tensorflow.python.keras.preprocessing.image import load_img,img_to_array
import numpy as np
import h5py
import heapq
from matplotlib import pyplot as plt

def getListMaxNumIndex(num_list,topk=3):
    '''
    获取列表中最大的前n个数值的位置索引
    '''
    # max_num_index=map(num_list.index, heapq.nlargest(topk,num_list))
    min_num_index=map(num_list.index, heapq.nsmallest(topk,num_list))
    min_index_list = [x for x in min_num_index]
    return min_index_list

def load_image_value(image_dir):
    image = load_image(image_dir)
    image = np.expand_dims(image, axis=0)
    return image

def load_image(image_dir):
    image = load_img(image_dir, target_size=(224, 224))
    image = img_to_array(image)
    image = image / 255
    return image

def show_result(retreival_image_dir, root_dir, model, std_feature_list, std_dir_list, topk):
    retreival_image = load_image_value(retreival_image_dir)
    retreival_id = (retreival_image_dir.split('#',1)[0]).rsplit('\\',1)[1]
    feature, _, __ = model.predict_on_batch([retreival_image, retreival_image, retreival_image])
    distance_list = np.sum(np.square(std_feature_list - feature), axis=1).tolist()
    index_list = getListMaxNumIndex(distance_list, topk)
    plt.figure()
    plt.imshow(load_image(retreival_image_dir))
    plt.title('search')
    plt.axis('off')
    plt.show()
    figure = plt.figure()
    for i in range(0, len(index_list)):
        plt.subplot(5,4,i+1)
        plt.subplots_adjust(left=0.0, right=0.25,bottom=0.0, top=0.2, wspace=0.0, hspace=0.0)
        # plt.tight_layout()
        std_dir = std_dir_list[index_list[i]]
        id = (std_dir.split('#',1)[0]).rsplit('\\',1)[1]
        if retreival_id == id:
            id = 'pick'
        plt.imshow(load_image(root_dir + std_dir))
        plt.title(id)
        plt.axis('off')
    plt.show()




ROOT_DIR = r'G:\毕设资料\dataset'
FEATURE_HDF5 = h5py.File('scripts/hdf5/std_feature.h5', 'r')
std_feature_list = FEATURE_HDF5['std_feature']
std_dir_list = FEATURE_HDF5['std_dir']
print(std_feature_list.shape)
model = create_model()
model.load_weights('my_model_weights.h5')
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9))

dir = r'G:\毕设资料\dataset\N1单人沙发\3FO4K1LSXY7Q#欧式印花布艺单人沙发带银边#600x450_33.jpg'
show_result(dir, ROOT_DIR, model, std_feature_list, std_dir_list, 20)