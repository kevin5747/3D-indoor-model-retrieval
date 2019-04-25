import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=15)
data1=[0.1784,0.3643,0.4171]
data2=[0.2532,0.4857,0.5579]
data3=[0.3426,0.5924,0.6915]
data4=[0.7581,0.9243,0.9874]
x=[1,5,10]
print(plt.style.available)

# plt.style.use("classic")
plt.figure()
plt.rcParams['font.sans-serif']=['SimHei']#设置所有中文以这个字体显示
plt.gca().spines['top'].set_visible(False)  # 去掉上边框
plt.gca().spines['right'].set_visible(False)  # 去掉右边框
plt.grid(linestyle="--")  # 设置背景网格线为虚线
plt.plot(x, data1, 'ro--', label=r'文献[20]',)
plt.plot(x, data2, 'gv--', label=r'文献[20]+图像主体检测')
plt.plot(x, data3, 'y^--', label=r'本文方法')
plt.plot(x, data4, 'b<--', label=r'本文方法+图像主体检测')

#在点上显示数据
for a, b in zip(x, data1):
    plt.text(a, b, b, ha='center', va='bottom')
for a, b in zip(x, data2):
    plt.text(a, b, b, ha='center', va='bottom')
for a, b in zip(x, data3):
    plt.text(a, b, b, ha='center', va='bottom')
for a, b in zip(x, data4):
    plt.text(a, b, b, ha='center', va='bottom')

plt.xlim((0, 11))
plt.ylim((0, 1))
plt.xticks(range(1,11))
plt.grid(True)
plt.xlabel('Top-k',fontsize=13)
plt.ylabel(r'命中率',fontsize=13)
plt.legend(loc="lower right")
plt.savefig('./Top-k命中率.svg', format='svg')
plt.show()
