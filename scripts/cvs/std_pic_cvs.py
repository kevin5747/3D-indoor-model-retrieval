import os
import pandas as pd
import random
ROOT_DIR = r'G:\毕设资料\dataset\N1单人沙发'

imageList = []
list = os.listdir(ROOT_DIR)
for i in range(0,len(list)):
    if(list[i].endswith('1000x1000.jpg')):
        imageList.append('\\N1单人沙发' + '\\' + list[i])

save = pd.DataFrame({'std_pic':imageList})
save.to_csv('std_pic.csv',index=False,sep='$',encoding='utf-8')