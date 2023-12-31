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

# 改进版softmax。用户输出设计
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

# P73页
def get_data():
    '''
    第 1 个参数 normalize,设置是否将输入图像正规化为0.0~1.0的值。如果将该参数设置为False, 则输入图像的像素会保持原来的0~255。
    第 2 个参数 flatten,  设置是否展开输入图像(变成一维数组)。如果将该参数设置为False, 则输入图像为1 × 28 × 28的三维数组; 若设置为True, 则输入图像会保存为由784个元素构成的一维数组。
    第 3 个参数 one_hot_label,设置是否将标签保存为one-hot表示(one-hot representation)。one-hot表示是仅正确解标签为1, 其余皆为0的数组, 就像[0,0,1,0,0,0,0,0,0,0]这样。
               当one_hot_label为False时, 只是像7、2这样简单保存正确解标签; 
               当one_hot_label为True时, 标签则保存为one-hot表示
    '''
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=False, flatten=True, one_hot_label=False)
    return x_test, t_test

def test1():
    x, t = get_data()
    print(t)
    print(x.shape)
    print(t.shape)
    for tt in range(len(x)):
        if x[0][tt] != 0:
            print(tt)
            print(x[0][tt])
            break
    print(x[0][203])
    print(t[0])

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

def neuralnet1():
    x, t = get_data()
    network = init_network()

    accuracy_cnt = 0
    for i in range(len(x)):
        y = predict(network, x[i])
        # print(y.shape)
        p = np.argmax(y)
        # print(p.shape)
        if p == t[i]:
            accuracy_cnt += 1
    print("Accuracy:" + str(float(accuracy_cnt) / len(x)))

# neuralnet1()

def neuralnet_patch():
    x, t = get_data()
    network = init_network()
    
    batch_size = 100
    accuracy_cnt = 0
    
    for i in range(0, len(x), batch_size):
        x_batch = x[i:i+batch_size]
        y_batch = predict(network, x_batch)
        p = np.argmax(y_batch, axis=1)
        accuracy_cnt += np.sum(p == t[i:i+batch_size])
    
    print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
# neuralnet_patch()