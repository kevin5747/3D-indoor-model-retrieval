import os
import pandas as pd
import random
imageList=[]
rootDir = r'D:\dataset\N1单人沙发'
list = os.listdir(rootDir)
for i in range(0,len(list)):
    if(list[i].endswith('1000x1000.jpg')):
        imageList.append('\\N1单人沙发' + '\\' + list[i])

positiveList = []
anchorList = []
negativeList = []
for image in list:
    if(image.endswith('1000x1000.jpg')):
        continue
    anchorList.append('\\N1单人沙发' + '\\' + image)

random.shuffle(anchorList)

for image in anchorList:
    subStr = image.rsplit('#',1)[0]
    positiveList.append(subStr + '#1000x1000.jpg')

for image in positiveList:
    negative = random.choice(imageList)
    while(negative == image):
        negative = random.choice(imageList)
    negativeList.append(negative)



save = pd.DataFrame({'anchor':anchorList,'positive':positiveList,'negative':negativeList})
save.to_csv('mydata.csv',index=False,sep='$',encoding='utf-8')


# for image in imageList:
#     for i in range(0,36):
#         subStr = image.split('.jpg')[0]
#         positiveList.append(image)

