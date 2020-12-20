import scipy
from liblinearutil import *
import numpy as np
import math

train_filename = 'hw4_train.txt'
test_filename = 'hw4_test.txt'


def secondTransform(datalist):  # second-order polynomial transformation
    output = [1]
    for item in datalist:
        output.append(item)
    for i in range(len(datalist)):
        for j in range(i, len(datalist)):
            output.append(datalist[i]*datalist[j])
    return output


def dataSet(filename):
    f = open(filename, 'r')
    data = []
    label = []
    for line in f:
        content = []
        y = 0
        for index, element in enumerate(line.split()):
            if index != 6:
                content.append(float(element))
            else:
                y = float(element)
        data.append(secondTransform(content))
        label.append(y)
    return np.array(data), np.array(label)


trainX, trainY = dataSet(train_filename)
testX, testY = dataSet(test_filename)


def cost(lamda):  # C 與 Lamda 關係
    return 1/(2*lamda)


def Prob16():
    lamdaList = [10**(-4), 10**(-2), 10**(0), 10**(2), 10**(4)]
    for lamda in lamdaList:
        print("Lamda: ", lamda)
        stringC = str(cost(lamda))
        command = '-s 0 -c ' + stringC + ' -e 0.000001 -q'
        m = train(trainY, trainX, command)
        p_label, p_acc, p_val = predict(testY, testX, m)
        print("================================")


def Prob17():
    lamdaList = [10**(-4), 10**(-2), 10**(0), 10**(2), 10**(4)]
    for lamda in lamdaList:
        print("Lamda: ", lamda)
        stringC = str(cost(lamda))
        command = '-s 0 -c ' + stringC + ' -e 0.000001 -q'
        m = train(trainY, trainX, command)
        p_label, p_acc, p_val = predict(trainY, trainX, m)
        print("================================")


def Prob18():
    lamdaList = [10**(-4), 10**(-2), 10**(0), 10**(2), 10**(4)]
    for lamda in lamdaList:
        print("Lamda: ", lamda)
        stringC = str(cost(lamda))
        command = '-s 0 -c ' + stringC + ' -e 0.000001 -q'
        m = train(trainY[:120], trainX[:120, :], command)
        p_label, p_acc, p_val = predict(trainY[120:], trainX[120:, :], m)
        print("================================")
    print("The best lamda is 0.01")
    bestC = str(cost(0.01))
    bestCommand = '-s 0 -c ' + bestC + ' -e 0.000001 -q'
    m = train(trainY[:120], trainX[:120, :], bestCommand)
    p_label, p_acc, p_val = predict(testY, testX, m)


def Prob19():
    print("The best lamda from previous problem is 0.01")
    bestC = str(cost(0.01))
    bestCommand = '-s 0 -c ' + bestC + ' -e 0.000001 -q'
    m = train(trainY, trainX, bestCommand)
    p_label, p_acc, p_val = predict(testY, testX, m)


def Prob20():
    lamdaList = [10**(-4), 10**(-2), 10**(0), 10**(2), 10**(4)]
    for lamda in lamdaList:
        print("Lamda: ", lamda)
        stringC = str(cost(lamda))
        command = '-s 0 -c ' + stringC + ' -e 0.000001 -v 5 -q'
        m = train(trainY, trainX, command)
        print("================================")


if __name__ == '__main__':
    # Prob16()
    # Prob17()
    # Prob18()
    # Prob19()
    Prob20()
