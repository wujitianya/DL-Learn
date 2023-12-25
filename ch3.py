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

def identity_function(x):
    return x

'''用于演示神经网络利用numpy的实现'''
def form3_9():
    X = np.array([1.0, 0.5])
    W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    B1 = np.array([0.1, 0.2, 0.3])
    print(X.shape)
    print(W1.shape)
    print(B1.shape)

    A1 = np.dot(X, W1) + B1
    print(A1)
    
    Z1 = sigmoid(A1)
    print(Z1)

    W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    B2 = np.array([0.1, 0.2])
    print(Z1.shape) # (3,)
    print(W2.shape) # (3, 2)
    print(B2.shape) # (2,)
    A2 = np.dot(Z1, W2) + B2
    Z2 = sigmoid(A2)
    print(Z2)

    W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
    B3 = np.array([0.1, 0.2])
    A3 = np.dot(Z2, W3) + B3
    Y = identity_function(A3) # 或者Y = A3
    print(Y)
# form3_9()

def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])
    return network

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)
    return y
network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
# print(y) # [ 0.31682708 0.69627909]

# test 笔记本 —— 已完成

# 改进版softmax
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

import sys, os
sys.path.append(os.pardir) #为了导入父目录中的文件而进行的设定
from dataset.mnist import load_mnist
import numpy as np
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

# (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

'''
img = x_train[16]
label = t_train[16]
print(label)
print(img.shape)

img = img.reshape(28, 28)
img_show(img)
'''
'''
cc = 0
for i in range(len(img)):
    if img[i] > 0:
        print(i, ":", img[i])
        cc += 1
print("sum is:", cc)
'''
'''
print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)
'''

print("this is ch3.py")
