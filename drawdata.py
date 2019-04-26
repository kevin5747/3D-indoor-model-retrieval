import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=15)
data1=[17.84,36.43,41.71]
data2=[25.32,48.57,55.79]
data3=[34.26,59.24,69.15]
data4=[75.81,92.43,98.74]
x=[1,5,10]
print(plt.style.available)

# plt.style.use("classic")
plt.figure()
plt.rcParams['font.sans-serif']=['SimHei']#设置所有中文以这个字体显示
plt.gca().spines['top'].set_visible(False)  # 去掉上边框
plt.gca().spines['right'].set_visible(False)  # 去掉右边框
plt.grid(linestyle="--")  # 设置背景网格线为虚线
plt.plot(x, data1, 'ro--', label=r'文献[20]方法',)
plt.plot(x, data2, 'rv-', label=r'文献[20]方法+图像主体检测')
plt.plot(x, data3, 'bD--', label=r'本文方法')
plt.plot(x, data4, 'bs-', label=r'本文方法+图像主体检测')

#在点上显示数据
for a, b in zip(x, data1):
    plt.text(a, b, b, ha='right', va='bottom')
for a, b in zip(x, data2):
    plt.text(a, b, b, ha='right', va='bottom')
for a, b in zip(x, data3):
    plt.text(a, b, b, ha='right', va='bottom')
for a, b in zip(x, data4):
    plt.text(a, b, b, ha='right', va='bottom')

plt.xlim((0, 11))
plt.ylim((0, 100))
plt.xticks(range(1,11))
plt.grid(True)
plt.xlabel('Top-k',fontsize=13)
plt.ylabel(r'命中率(%)',fontsize=13)
plt.legend(loc="lower right")
plt.savefig('./Top-k命中率.svg', format='svg',bbox_inches='tight')
plt.show()
