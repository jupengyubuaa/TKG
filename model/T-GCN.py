# -*- coding=utf-8-*-
# @Time: 2022/12/20 15:19
# @Author: 鞠鹏羽
# @File: T-GCN.py
# @Software: PyCharm

# 收集前后邻居，根据已有的筛选机制，需要改变窗口的方向
'''
与environment的get_padding_actions

'''
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# 时间感知采样
# GCN聚合
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from random import random

def expmove(y,alpha=0.6):
    n=len(y)
    M=np.zeros(n)# 生成空序列，用于存储指数平滑值M
    M[0]=y[0]# 初始指数平滑值的赋值
    for i in range(1,len(y)):
        M[i]=alpha*y[i-1]+(1-alpha)*M[i-1]# 开始预测
    return M
# 一次指数平滑序列
def SES(y,alpha=0.6):
    ss1 = expmove(y,alpha)
    # 二次指数平滑序列
    ss2 = expmove(ss1,alpha)
    y_pred=np.zeros(len(y))

    for i in range(1,len(y)):
        y_pred[i]=2*ss1[i-1]-ss2[i-1]+alpha/(1-alpha)*(ss1[i-1]-ss2[i-1])
    # 2023
    y_pred=2*ss1[-1]-ss2[-1]+alpha/(1-alpha)*(ss1[-1]-ss2[-1])*1
    return y_pred
# 预测原时间序列


def calc_next_s(alpha, x):
    s = [0 for  i in range(len(x))]
    s[0] = np.sum(x[0:3]) / float(3)
    for i in range(1, len(s)):
        s[i] = alpha*x[i] + (1-alpha)*s[i-1]
    return s

# 预测
alpha=0.9
def time_predict(x):
    s1 = calc_next_s(alpha,x)
    s2 = calc_next_s(alpha,s1)
    s3 = calc_next_s(alpha, s2)
    a3 = [(3 * s1[i] - 3 * s2[i] + s3[i]) for i in range(len(s3))]
    b3 = [((alpha / (2 * (1 - alpha) ** 2)) * ((6 - 5 * alpha) * s1[i] - 2 * (5 - 4 * alpha) * s2[i] + (4 - 3 * alpha) * s3[i])) for i in range(len(s3))]
    c3 = [(alpha ** 2 / (2 * (1 - alpha) ** 2) * (s1[i] - 2 * s2[i] + s3[i])) for i in range(len(s3))]
    pred = a3[-1]+b3[-1]*1+c3[-1]*(1**2)
    return pred
print(SES([1,2,3,4],0.6))
print(time_predict([1,2,3,4]))
# y=np.random.random(10)
# n=9
# a=0.6
# print(y)
# print(SES(y,n,a))
#
# data=[random() for x in range(1,100)]
# model=AutoReg(data,lags=1)
# model_fit=model.fit()
# yhat=model_fit.predict(len(data),len(data))
# print(yhat)
# plt.figure(figsize=(16,8))
# plt.plot(data,label='train')
# plt.plot(yhat,'o')
# plt.legend(loc='best')
# plt.show()
# model = ARIMA(data, order=(0, 0, 1))
# model_fit = model.fit()
# yhat=model_fit.predict(len(data),len(data))
# print(yhat)
#
# model = ARIMA(data, order=(2, 0, 1))
# model_fit = model.fit()
# yhat=model_fit.predict(len(data),len(data))
# print(yhat)
# model = ARIMA(data, order=(1, 1, 1))
# model_fit = model.fit()
# yhat = model_fit.predict(len(data), len(data), typ='levels')
# print(yhat)
# model = SimpleExpSmoothing(data)
# model_fit = model.fit()
# # make prediction
# yhat = model_fit.predict(len(data), len(data))
# print(yhat)
# model = ExponentialSmoothing(data)
# model_fit = model.fit()
# # make prediction
# yhat = model_fit.predict(len(data), len(data))
# print(yhat)
print(torch.eq(torch.ones(1),torch.zeros(1)))
a = torch.Tensor([[1, 2, np.nan], [2, np.nan, 4], [3, 4, 5]])

a = torch.where(torch.isnan(a), torch.full_like(a, 0), a)
a=np.random.rand(10)*10
a=torch.tensor(a)
print(torch.softmax(a,0))
print(a)
print(torch.eq(torch.Tensor([[1],[1],[1]]),torch.Tensor([[1,2,3],[1,2,3],[1,2,3]])))
print(torch.logical_xor(torch.tensor([True,False,True]),torch.tensor([False,False,True])))
print(torch.__version__)

# time_window=10
# a = self.entity_frequency[action_space[:, :, 1], query_timestamps.unsqueeze(-1).repeat(1, action_num)]
# # a的维度 batch_size,action_num,
# # 增加维度
# a.unsqueeze(-1).repeat(2,time_window)
# a=np.zeros(batch_size,action_num,time_window)
# for i in range(time_window):
#     a[:,:,i]=self.entity_frequency[action_space[:, :, 1], (query_timestamps-24*10+i*24).unsqueeze(-1).repeat(1, action_num)]
# b=np.zeros(batch_size,action_num,1)
# for i in range(time_window)
#     b[:,:,i]=SES(a[:,:,i],10,0.8)
# score=b.squeeze(-1)

print(type(a))
x=torch.rand(3)
print(x-24)
print(np.add(a,b))
a=np.array(a)
print(a)
print(0.1*a)

