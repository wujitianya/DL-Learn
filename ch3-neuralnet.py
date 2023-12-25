import numpy as np
from PIL import Image
import sys, os
import pickle
sys.path.append(os.pardir) #为了导入父目录中的文件而进行的设定
from dataset.mnist import load_mnist

# import ch3

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# from csdn，解决溢出问题
def sigmoid2(x):
    x_ravel = x.ravel()
    length = len(x_ravel)
    y = []
    for index in range(length):
        if x_ravel[index] >= 0:
            y.append(1.0 / (1 + np.exp(-x_ravel[index])))
        else:
            y.append(np.exp(x_ravel[index]) / (np.exp(x_ravel[index]) + 1))
    return np.array(y).reshape(x.shape)

# 改进版softmax
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=False, flatten=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid2(a1)
    # z1 = logistic_function(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid2(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

print("this is ch3-neuralnet.py")

x, t = get_data()
network = init_network()

accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y)
    if p == t[i]:
        accuracy_cnt += 1
print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
