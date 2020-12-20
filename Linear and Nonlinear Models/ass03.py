import numpy as np
import math

train_filename = 'hw3_train.txt'
test_filename = 'hw3_test.txt'


def dataSet(filename, x0=1):
    f = open(filename, 'r')
    data = []
    label = []
    for line in f:
        content = [x0]
        y = 0
        for index, element in enumerate(line.split()):
            if index != 10:
                content.append(float(element))
            else:
                y = 2 * float(element)-3
        data.append(np.array(content))
        label.append(y)
    return np.array(data), np.array(label)


x, y = dataSet(train_filename)
testX, testY = dataSet(test_filename)


def nonLinearDataSet(filename, Q, x0=1):
    f = open(filename, 'r')
    data = []
    label = []
    for line in f:
        content = [x0]
        y = 0
        for k in range(Q):
            for index, element in enumerate(line.split()):
                if index != 10:
                    content.append(float(element)**(k+1))
                else:
                    y = float(element)
        data.append(np.array(content))
        label.append(y)
    return np.array(data), np.array(label)


def linearRegression(x, y):
    w = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y.T)
    return w


def binaryError(w, x, y):
    N = len(y)
    error = 0
    for i in range(N):
        if w.T.dot(x[i]) * y[i] <= 0:
            error = error + 1
    return error/N


def SqrEin(w, x, y):
    error = 0
    N = len(y)
    for i in range(N):
        error = error+(w.T.dot(x[i])-y[i])**2
    return error/N


def Prob14():
    w_lin = linearRegression(x, y)
    Ein = SqrEin(w_lin, x, y)
    print(Ein)
    return Ein


def theta(s):
    return 1/(1+math.exp(-s))


def crossEntropyError(w, x, y):
    error = 0
    N = len(y)
    for i in range(N):
        error = error + math.log(1+math.exp(-y[i] * w.T.dot(x[i])))
    return error/N


def SGDfor15(x, y, lr=0.001):
    w = np.zeros(11)
    Ein = 1
    #w_lin = linearRegression(x, y)
    #linSqrEin = SqrEin(w_lin, x, y)
    linSqrEin = 0.60532  # 根據前題答案
    count = 0
    N = len(y)
    while Ein > 1.01*linSqrEin:
        pickNumber = np.random.randint(N)
        w = w + lr * theta(-y[pickNumber]*w.T.dot(x[pickNumber])) * \
            2*(y[pickNumber]-w.T.dot(x[pickNumber])) * x[pickNumber]
        Ein = SqrEin(w, x, y)
        count = count+1
    return count


def Prob15():
    k = 0
    number = 1000
    for i in range(number):
        number = SGDfor15(x, y)
        k = k + number
        print(i, number, (k/(i+1)))
    return(k/number)


def SGDforIter(x, y, w=np.zeros(11), lr=0.001):
    N = len(y)
    for i in range(500):
        pickNumber = np.random.randint(N)
        w = w + lr * theta(-y[pickNumber]*w.T.dot(x[pickNumber])
                           ) * y[pickNumber] * x[pickNumber]
        Ein = crossEntropyError(w, x, y)
    return Ein


def Prob16():
    err = 0
    for i in range(1000):
        number = SGDforIter(x, y)
        err = err + number
        print(i, number)
    return err/1000


def Prob17():
    w_lin = linearRegression(x, y)
    err = 0
    for i in range(1000):
        number = SGDforIter(x, y, w_lin)
        err = err + number
        print(i, number)
    return err/1000


def Prob18():
    w_lin = linearRegression(x, y)
    Ein = binaryError(w_lin, x, y)
    Eout = binaryError(w_lin, testX, testY)
    print(abs(Ein-Eout))
    return abs(Ein-Eout)


def Prob19():
    nlx, nly = nonLinearDataSet(train_filename, 3)
    testnlX, testnlY = nonLinearDataSet(test_filename, 3)
    nlw = linearRegression(nlx, nly)
    Ein = binaryError(nlw, nlx, nly)
    Eout = binaryError(nlw, testnlX, testnlY)
    print(abs(Ein-Eout))


def Prob20():
    nlx, nly = nonLinearDataSet(train_filename, 10)
    testnlX, testnlY = nonLinearDataSet(test_filename, 10)
    print(nlx[0])
    nlw = linearRegression(nlx, nly)
    Ein = binaryError(nlw, nlx, nly)
    Eout = binaryError(nlw, testnlX, testnlY)
    print(abs(Ein-Eout))

# Prob14()
# Prob15()
# Prob16()
# Prob17()
# Prob18()
# Prob19()
# Prob20()
