# -*- coding: utf-8 -*-

###########################################################################################
# Implementation of CURE (Clustering Using Representatives) Clustering Algorithm
# Author for codes: Chu Kun(kun_chu@outlook.com)
# Paper: https://www.sciencedirect.com/science/article/pii/S0306437901000084
# Reference: https://github.com/Kchu/CURE-cluster-python
###########################################################################################

import numpy as np
import scipy.spatial.distance as distance
import sys


# 返回两个向量之间的距离
def dist(vecA, vecB):
    return np.sqrt(np.power(vecA - vecB, 2).sum())


# 此类描述 CURE 聚类的数据结构和操作方法。
class CureCluster:
    def __init__(self, id__, center__):
        # 该簇对应的数据集
        self.points = center__
        # 该簇对应的代表点
        self.repPoints = center__
        # 该簇对应的质心
        self.center = center__
        # 该簇对应数据的索引
        self.index = [id__]

    def __repr__(self):
        return "Cluster " + " Size: " + str(len(self.points))

    def computeCentroid(self, clust):
        """
        根据其点计算并存储该集群的质心
        :param clust: 另一个簇
        :return:
        """
        totalPoints_1 = len(self.index)
        totalPoints_2 = len(clust.index)
        self.center = (self.center * totalPoints_1 + clust.center * totalPoints_2) / (totalPoints_1 + totalPoints_2)

    def generateRepPoints(self, numRepPoints, alpha):
        """
        计算并存储该集群的代表点
        :param numRepPoints: 代表点的个数
        :param alpha: 收缩因子
        :return:
        """
        # 存储着已经计算出的代表点
        tempSet = None
        for i in range(1, numRepPoints + 1):
            # 最大的距离
            maxDist = 0
            # 最大距离对应的点
            maxPoint = None
            for p in range(0, len(self.index)):
                # 如果计算第一个中心点，则选择距离质心最远的点作为第一个代表点
                # 如果不是第一个点，这个点距离代表点的距离是，离这个点最近的代表点的距离
                #     从所有点选择距离最远的点作为代表点
                # 代表点可以重复
                if i == 1:
                    minDist = dist(self.points[p, :], self.center)
                else:
                    X = np.vstack([tempSet, self.points[p, :]])
                    tmpDist = distance.pdist(X)
                    minDist = tmpDist.min()
                if minDist >= maxDist:
                    maxDist = minDist
                    maxPoint = self.points[p, :]
            if tempSet is None:
                tempSet = maxPoint
            else:
                tempSet = np.vstack((tempSet, maxPoint))
        # 通过alpha参数收缩并更新代表点
        for j in range(len(tempSet)):
            if self.repPoints is None:
                self.repPoints = tempSet[j, :] + alpha * (self.center - tempSet[j, :])
            else:
                self.repPoints = np.vstack((self.repPoints, tempSet[j, :] + alpha * (self.center - tempSet[j, :])))

    # 通过簇的代表点计算并存储此簇与另一个簇之间的距离。
    # 距离为最近的代表点的距离
    def distRep(self, clust):
        distRep = float('inf')
        for repA in self.repPoints:
            # clust刚被初始化的时候，repPoint不是二维的
            if type(clust.repPoints[0]) != list:
                repB = clust.repPoints
                distTemp = dist(repA, repB)
                if distTemp < distRep:
                    distRep = distTemp
            else:
                for repB in clust.repPoints:
                    distTemp = dist(repA, repB)
                    if distTemp < distRep:
                        distRep = distTemp
        return distRep

    # 将此簇与给定簇合并，重新计算质心和代表点。
    def mergeWithCluster(self, clust, numRepPoints, alpha):
        # 计算新的质心
        self.computeCentroid(clust)
        # 合并数据集
        self.points = np.vstack((self.points, clust.points))
        # 合并索引
        self.index = np.append(self.index, clust.index)
        # 重新计算代表点
        self.repPoints = None
        self.generateRepPoints(numRepPoints, alpha)


def runCURE(data, numRepPoints, alpha, numDesCluster):
    """
    描述CURE算法的过程
    :param data: 加载的数据集
    :param numRepPoints: 代表点的个数
    :param alpha: 收缩因子
    :param numDesCluster: 期望的簇的个数
    :return:
    """
    Clusters = []
    numCluster = len(data)
    numPts = len(data)
    # 初始化簇的距离矩阵，初始值为inf
    distCluster = np.ones([len(data), len(data)])
    distCluster = distCluster * float('inf')
    for idPoint in range(len(data)):
        # 为每一个点初始化类
        newClust = CureCluster(idPoint, data[idPoint, :])
        # 添加到Clusters列表中
        Clusters.append(newClust)
    # 计算点之间的距离
    for row in range(0, numPts):
        for col in range(0, row):
            distCluster[row][col] = dist(Clusters[row].center, Clusters[col].center)
    # 当目前簇的数量大于期望的数量，需要继续合并
    while numCluster > numDesCluster:
        if np.mod(numCluster, 50) == 0:
            print('Cluster count:', numCluster)

        # 找到距离最近的两个簇
        minIndex = np.where(distCluster == np.min(distCluster))
        minIndex1 = minIndex[0][0]
        minIndex2 = minIndex[1][0]

        # 将两个簇进行合并
        Clusters[minIndex1].mergeWithCluster(Clusters[minIndex2], numRepPoints, alpha)
        # 更新距离矩阵
        for i in range(0, minIndex1):
            distCluster[minIndex1, i] = Clusters[minIndex1].distRep(Clusters[i])
        for i in range(minIndex1 + 1, numCluster):
            distCluster[i, minIndex1] = Clusters[minIndex1].distRep(Clusters[i])
        # 删除被合并簇所在的行和列
        distCluster = np.delete(distCluster, minIndex2, axis=0)
        distCluster = np.delete(distCluster, minIndex2, axis=1)
        del Clusters[minIndex2]
        numCluster = numCluster - 1

    print('Cluster count:', numCluster)
    # 生成label
    Label = [0] * numPts
    for i in range(0, len(Clusters)):
        for j in range(0, len(Clusters[i].index)):
            Label[Clusters[i].index[j]] = i + 1

    return Label
