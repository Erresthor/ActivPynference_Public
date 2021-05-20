# -*- coding: utf-8 -*-
"""
Created on Mon May 17 17:23:53 2021

@author: cjsan
"""

import matplotlib.pyplot as plt
import numpy as np

def basic_autoplot(x):
    n = x.shape[0]
    plt.plot(np.linspace(1,n,n),x)
    plt.show()

#x = np.random.randint(0,10,(25,))

#basic_autoplot(x)