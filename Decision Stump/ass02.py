import numpy as np
import random


def data(N, tau=0):  # 依題目切割 x [-1,1]，並設定錯誤率 tau 使得 y flip
    x = np.linspace(-1, 1, N)
    y = np.zeros(N)
    for index, content in enumerate(x):
        k = 1
        if random.random() <= tau:
            k = -1
        if content > 0:
            y[index] = 1 * k
        else:
            y[index] = -1 * k
    dataArray = np.array([x, y])
    return dataArray.T


def randomPickedExample(allData, N, number, seed=None):  # 從所有資料點中進行抽取
    np.random.seed(seed)
    pickarray = []
    arrayIndex = np.sort(np.random.randint(N, size=number))
    for i in arrayIndex:
        pickarray.append(allData[i])
    return arrayIndex, np.array(pickarray)


def thetaArray(pickedarray):  # 依題目要求依題目要求記錄所有可能的 theta
    theta = [-1]
    for index in range(len(pickedarray)-1):
        middle = (pickedarray[index][0]+pickedarray[index+1][0])/2
        theta.append(middle)
    return np.array(theta)


def Ein(number, thetaArray, pickedarray):  # 計算Ein
    sList = [1, -1]
    Ein_record = {"s": 1, "theta": 1, "Ein": 1}
    for s in sList:
        for theta in thetaArray:
            count = 0
            for x, y in pickedarray:
                if s * (x-theta) == 0 and y > 0:
                    count = count + 1
                elif s * (x-theta)*y < 0:
                    count = count + 1
                else:
                    pass
            if Ein_record["Ein"] > count/number:
                Ein_record["s"] = s
                Ein_record["theta"] = theta
                Ein_record["Ein"] = count/number
            elif Ein_record["Ein"] == count/number and Ein_record["s"]+Ein_record["theta"] > s + theta:
                Ein_record["s"] = s
                Ein_record["theta"] = theta
            else:
                pass
    return Ein_record


def Eout(EinDict, N, number, arrayIndex, allData):  # 計算Eout
    count = 0
    for i in range(N):
        if i not in arrayIndex:
            x, y = allData[i]
            if EinDict["s"] * (x-EinDict["theta"]) == 0 and y > 0:
                count = count + 1
            elif EinDict["s"] * (x-EinDict["theta"]) * y < 0:
                count = count + 1
            else:
                pass
    return count/(N-number)


def main(N, tau=0, number=2):
    allData = data(N, tau)
    arrayIndex, pickedArray = randomPickedExample(allData, N, number)
    theta = thetaArray(pickedArray)
    Ein_record = Ein(number, theta, pickedArray)
    Eout_value = Eout(Ein_record, N, number, arrayIndex, allData)
    return Eout_value-Ein_record["Ein"]


if __name__ == "__main__":
    time = 10000
    N = 1000
    tau = 0.1
    number = 2
    deltaE = 0
    for i in range(time):
        deltaE = deltaE + main(N, tau, number)
    print("Total number:" + str(N))
    print("Tau:" + str(tau))
    print("Sample number:" + str(number))
    print(deltaE/time)
