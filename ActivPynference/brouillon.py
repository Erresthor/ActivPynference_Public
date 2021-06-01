# -*- coding: utf-8 -*-
"""
Created on Fri May 21 11:13:30 2021

@author: cjsan
"""
import math as m
import numpy as np
import matplotlib.pyplot as plt


def da_big_poto_sum(x,N):
    for i in range(N):
        x = m.sqrt(x)
    print(x)

n = 50
X = np.linspace(0,50,n)
L = np.zeros((n,))
for i in range (n):
    L[i] =  da_big_poto_sum(X[i],18000)

plt.plot(X,L)
plt.show()