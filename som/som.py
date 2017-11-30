# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 18:48:25 2017

@author: dell
"""


import numpy as np
from numpy.linalg import cholesky
import math

#初始化输入输出层链接权值  
def initCompetition(n,m,d):
    array = np.random.random(size = n*m*d)
    com_weight = array.reshape(n,m,d)
    return com_weight
#计算欧氏距离
def cal2NF(data):
    res = 0
    for x in data:
        res = res + x*x
    return res**0.5
#将数据归一化
def normalize_data(train_data):
    for data in train_data:
        two_NF = cal2NF(data)
        for i in range(len(data)):
            data[i] = data[i] / two_NF
    return train_data
#将权值归一化
def nomalize_weight(com_weight):
    for x in com_weight:
        for data in x:
            two_NF = cal2NF(data)
            for i in range(len(data)):
                data[i] = data[i] / two_NF
    return com_weight
#得到获胜神经元索引
def getWinner(data,com_weight):
    max_sim = 0
    mark_n = 0
    mark_m = 0
    n,m,d = np.shape(com_weight)
    for i in range(n):
        for j in range(m):
            if sum(data * com_weight[i,j]) > max_sim:
                max_sim = sum(data * com_weight[i,j])
                mark_n = i
                mark_m = j
    return mark_n,mark_m
#得到获胜神经元周围的兴奋神经元的索引
def getNeighbor(n,m,N_neighbor,com_weight):
    res = []
    nn,mm,cc = np.shape(com_weight)
    for i in range(nn):
        for j in range(mm):
            N = int(((i-n)**2 + (j-m)**2)**0.5)
            if N <= N_neighbor:
                res.append((i,j,N))
    return res
#学习率 与迭代次数和拓扑距离相关
def eta(t,N):
    return (0.3/(t+1))*(math.e**(-N))    

#som算法核心                  
def som(train_data,com_weight,T,N_neighbor):
    for t in range(T-1):
        com_weight = nomalize_weight(com_weight)
        for data in train_data:
            n,m = getWinner(data,com_weight)
            neighbor = getNeighbor(n,m,N_neighbor,com_weight)
            for x in neighbor:
                j_n = x[0]
                j_m = x[1]
                N = x[2]
                com_weight[j_n][j_m] = com_weight[j_n][j_m] + eta(t,N)*(data - com_weight[j_n][j_m])
            N_neighbor = N_neighbor - (t+1)/200
    res = {}
    kind = 0
    N,M,p = np.shape(com_weight)
    for i in range(len(train_data)):
        n,m = getWinner(train_data[i],com_weight)
        key = n * M + m
        if(res.has_key(key)):
            res[key].append(i)
        else:
            kind = kind + 1
            res[key] = []
            res[key].append(i)
#手动生成数据
            

#设置高斯分布的sigma
sigma1=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
sigma2=np.array([[0.81,0,0,0],[0,0.81,0,0],[0,0,0.81,0],[0,0,0,0.81]])
sigma3=np.array([[0.25,0,0,0],[0,0.25,0,0],[0,0,0.25,0],[0,0,0,0.25]])
sigma4=np.array([[1,0,0,0],[0,0.81,0,0],[0,0,0.81,0],[0,0,0,0.81]])

R1= cholesky(sigma1)
R2= cholesky(sigma2)
R3= cholesky(sigma3)
R4= cholesky(sigma4)
#设置高斯分布的mu
mu_1 = np.array([[4,4,4,20]])
mu_2 = np.array([[14,6,6,6]])
mu_3 = np.array([[8,1,8,8]])
mu_4 = np.array([[1,15,15,4]])
#生成4维高斯分布数据，
s_1 = np.dot(np.random.randn(500,4),R1)+mu_1
s_2 = np.dot(np.random.randn(500,4),R2)+mu_2
s_3 = np.dot(np.random.randn(500,4),R3)+mu_3
s_4 = np.dot(np.random.randn(500,4),R4)+mu_4
#将数组拼接 形成训练数据 
list_1 = []
for data in s_1:
    list_temp = list(data)
    list_1.append(list_temp)
for data in s_2:
    list_temp = list(data)
    list_1.append(list_temp)
for data in s_3:
    list_temp = list(data)
    list_1.append(list_temp)
for data in s_4:
    list_temp = list(data)
    list_1.append(list_temp)
train_data = np.array(list_1)

T = 50
N_neighbor = 10
train_data = normalize_data(train_data)
com_weight = initCompetition(3,3,4)
som(train_data,com_weight,T,N_neighbor)