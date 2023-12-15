
def myAND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1 * w1 + x2 * w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1
'''
r1 = myAND(0,0)# 输出0
r2 = myAND(1,0)
r3 = myAND(0,1)
r4 = myAND(1,1)
print("与门第一版：%d, %s, %d, %s"%(r1,r2,r3,r4))
'''

# 利用numpy的矩阵计算制作与门感应机 ---------------------------------------------------------------
import numpy as np 
x = np.array([0,1])
w = np.array([0.5,0.5])
b = -0.7
r1 = np.sum(w*x)
r2 = np.sum(w*x) + b
'''
print(x)
print(w)
print(r1)
print(r2)
'''
# 与
def myAND2(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1
# 与非 
def myNAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1
# 或  
def myOR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

# 异或
def myXOR(x1, x2):
    s1 = myNAND(x1, x2)
    s2 = myOR(x1, x2)
    y = myAND2(s1, s2)
    return y

