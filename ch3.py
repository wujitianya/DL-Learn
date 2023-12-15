'''
图3-4很重要，说明了激活函数的含义

***，如果将激活函数从阶跃函数换成其他函数，就可以进入神经网络的世界了。
'''

import numpy as np
import matplotlib.pylab as plt

def step_function(x):
    # return np.array(x>0, dtype=int)
    y = x > 0
    return y.astype(int)

def stepphysical():
    x = np.arange(-5.0, 5.0, 0.1)
    y = step_function(x)
    print(x)
    print("-----------------")
    print(y)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1) # 指定y轴的范围
    plt.show()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoidphysical():
    x = np.arange(-5.0, 5.0, 0.1)
    y = sigmoid(x)
    plt.plot(x,y)
    plt.ylim(-0.1, 1.1)
    plt.show()
    
# sigmoidphysical()
    
def relu(x):
    return np.maximum(0, x)
'''
x = np.array([[1,2],[3,4]])
y = np.array([[1,2],[3,4],[5,6]])
print(x*y)
#print(np.dot(x,y))
'''