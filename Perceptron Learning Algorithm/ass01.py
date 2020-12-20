import numpy as np
import random

filename = 'hw1_train.txt'

# Turn the data from file to array
def dataSet( x0 = 1 , scaleFactor = 1):
    f = open(filename, 'r')
    data=[]
    for line in f:
        insert_data=[] 
        content=[x0]
        y=0
        for index, element in enumerate(line.split()):
            if index != 10:
                content.append(scaleFactor * float(element))
            else:
                y = float(element)
        insert_data.append(tuple(content))
        insert_data.append(y)
        data.append(tuple(insert_data))
    return np.array(data)

# Update the weight 
def updateWeight( w, xy):
    new_weight_list = []
    x, y= xy
    for weight, x_content in zip(w,x):
        new_weight_list.append(weight + y*x_content)
    return tuple(new_weight_list)

# Decide the list of which data we will test
def randomPickedExample( number, seed=None):
    np.random.seed(seed)
    return np.random.randint( 100, size = 5*number)

# Check all the data by test the final weight
def checkAll( w, all):
    count = 0
    for item in all:
        if np.array(w).T.dot(item[0]) * item[1] > 0:
            pass
        else:
            count=count+1
    print("Bad number: " + str(count))
    

def PLA(x0 = 1, scale = 1):
    dataset = dataSet(x0,scale)
    N = len(dataset) # Number of total examples
    w = tuple(np.zeros(11)) # Initial weight
    pick_number = 5 * N # Number of testing example
    count = pick_number
    updatingCount = 0
    example_list = randomPickedExample(pick_number)
    while count:
        selectedData = dataset[example_list[pick_number-count]]
        if np.array(w).T.dot(selectedData[0]) == 0:
            if selectedData[1] < 0:
                count = count-1
            else: # Incorrect. Update weight and initialize count and pick examples again
                updatingCount+=1 
                #print("updating...")
                w = updateWeight( w, selectedData)
                count = pick_number
                example_list = randomPickedExample(pick_number)
        elif np.array(w).T.dot(selectedData[0])* selectedData[1] > 0:
            count = count-1 # Correct, and next
        else: # Incorrect. Update weight and initialize count and pick examples again
            updatingCount+=1
            #print("updating...")
            w = updateWeight( w, selectedData)
            count = pick_number
            example_list = randomPickedExample(pick_number)
    return w, updatingCount

def main(x0 = 1, scale = 1):
    upTime=[]
    w0List=[]
    for i in range(1000):
        w, number=PLA(x0, scale)
        upTime.append(number)
        w0List.append(w[0])
    upTime.sort()
    w0List.sort()
    print("====================================")
    print("When x0 = %d; scale = %d" %(x0,scale))
    print("Med # of update: " + str(upTime[499]))
    print("Med w0: " + str(w0List[499]))

if __name__ == '__main__':
    main(1,1)
    main(10,1)
    main(0,1)
    main(0,0.25)



