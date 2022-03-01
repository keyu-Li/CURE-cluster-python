# -*- coding:utf-8 -*-

from CURE import *
import sys, time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

# 代表点的数量
numRepPoints = 5
# 收缩因子
alpha = 0.1
# 期望的簇的个数
numDesCluster = 15

start = time.clock()
data_set = np.loadtxt('./R15.txt')
data = data_set[:, 0:2]
Label_true = data_set[:, 2]
print("等待CURE聚类完成...")
Label_pre = runCURE(data, numRepPoints, alpha, numDesCluster)
print("CURE完成聚类!!\n")
end = time.clock()
print("用时：", end - start, "\n")
# 计算NMI
nmi = metrics.v_measure_score(Label_true, Label_pre)
print("NMI =", nmi)
# 绘制结果
plt.subplot(121)
plt.scatter(data_set[:, 0], data_set[:, 1], marker='.')
plt.text(0, 0, "origin")
plt.subplot(122)
scatterColors = ['black', 'blue', 'green', 'yellow', 'red', 'purple', 'orange', 'brown', 'cyan', 'brown',
                 'chocolate', 'darkgreen', 'darkblue', 'azure', 'bisque']
for i in range(data_set.shape[0]):
    color = scatterColors[Label_pre[i] - 1]
    plt.scatter(data_set[i, 0], data_set[i, 1], marker='o', c=color)
plt.text(0, 0, "clusterResult")
plt.show()
# plt.savefig("Figure_1.png")
