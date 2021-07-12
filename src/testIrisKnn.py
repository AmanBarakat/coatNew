import numpy as np
import pandas as pd
import math
import time
import matplotlib.pyplot as plt
import sys
import statistics
import itertools

def poly(p,r,y,x):
    return pow(abs(r-abs(y-x)),p)/pow(r,p)

def calcSim(data,y,x,pol,p,w):
    s=0
    for i in range(len(w)) :
        s+=poly(p[i], pol[i], data[x][i],data[y][i])*w[i]

    return s/sum(w)
def calcOneSim(y,x,pol,p,w):
    s=0
    for i in range(len(w)) :
        s+=poly(p[i], pol[i], y[i],x[i])*w[i]
    return s/np.sum(w)

def getOut(x,y):
    if x==y:
        return 1
    else:
        return 0
        
def knn(simArr,train,n):
    dataOut = train['class'].values

    simSorted=np.argsort(simArr)
    # print(simArr)
    sumVersicolor=0
    sumVirginica=0
    sumSetosa=0
    # print(simSorted)
    simSortedDesc=simSorted[::-1][:n]
    for i in range(n):
        # print(f'{i} nearest neighbor is {dataOut[simSortedDesc[i]]}')
        if dataOut[simSortedDesc[i]]=='Iris-versicolor':
            sumVersicolor+=1
        elif dataOut[simSortedDesc[i]]=='Iris-virginica':
            sumVirginica+=1
        elif dataOut[simSortedDesc[i]]=='Iris-setosa':
            sumSetosa+=1
    maxClass=max(sumVersicolor,sumVirginica,sumSetosa)
    if maxClass==sumVersicolor:
        return 'Iris-versicolor'
    elif maxClass==sumVirginica:
        return 'Iris-virginica'
    else:
        return 'Iris-setosa'
def calculateAug(df,minDiff,simArr,t):
    col=df.shape[1] - 1
    outArr =[]
    data = df.values
    dataOut = df['class'].values
    for i in range(m):
        outArr.append(getOut(t[col],dataOut[i]))
    outArr.append(1)
    newInv=0
    while newInv <= minDiff:
        for x in range(m):
            simVal =  calcOneSim(data[x],t,pol,p,w)
            outVal =  getOut(dataOut[x],t[col])

            for y in range(m):
                if matrice_outcome[x][y] == outVal:
                    continue
                else:
                    if  matrice_outcome[x][y] > outVal:
                       if matrice_similarity[x][y] <= simVal: 
                           newInv+=1
                    elif matrice_similarity[x][y] >= simVal:
                        newInv+=1
        for a, b in itertools.combinations(enumerate(outArr), 2):
                if a[1] == b[1]:
                    continue
                else:
                    if a[1]>b[1]:
                        big=a[0]
                        small=b[0]
                    else:
                        big=b[0]
                        small=a[0]
                    if simArr[small] - simArr[big]>=0:
                        newInv+=1
        break
    return newInv
def compl(X,w,v=0):
    inv=0

    tst=0
    for x in range(len(matrice_outcome)):
        for a, b in itertools.combinations(enumerate(matrice_outcome[x]), 2):
            if a[1] == b[1]:
                continue
            else:
                if a[1]>b[1]:
                    big=a[0]
                    small=b[0]
                else:
                    big=b[0]
                    small=a[0]
                if matrice_similarity[x][small] - matrice_similarity[x][big]>=0:
                    inv+=1
                    if v!=0:
                        if inv>v:
                            return inv
    return inv
def predictAcc(test,train,c,w):
    total=0
    potential_outcomes=['Iris-versicolor','Iris-virginica','Iris-setosa']
    classOpt=''
    for index, row in test.iterrows():
        minDiff = (m*m) + (2*m)

        originalClass=row['class']
        # print(f'Classe originale {originalClass}')
        for r in potential_outcomes:
            rowCopy=row
            rowCopy['class']=r
            t=rowCopy.values


            simArr = []

            data = train.values


            for i in range(m):        
                simArr.append(calcOneSim(t,data[i],pol,p,w))
            classOpt=knn(simArr,train,5)
            # simArr.append(1)

            # aug=calculateAug(train,minDiff,simArr,t)

            # if aug<=minDiff:
            #     minDiff=aug
            #     classOpt=r
          
        if originalClass==classOpt:
            total+=1

    return total/test.shape[0]

if __name__ == '__main__':
    start = time.process_time()
 
    accuracies, complexities,  accuracySd,complexitiesSd,timePreds,timePredsSd=([] for i in range(6))
    weights=[[1, 1, 1, 1], [36, 24, 31, 86], [76, 44, 41, 60], [28, 46, 86, 38], [78, 97, 7, 72], [21, 58, 35, 24], [75, 34, 73, 33], [23, 40, 27, 56], [12, 94, 78, 81], [77, 33, 37, 18], [32, 28, 79, 21], [51, 72, 37, 46], [80, 86, 58, 2], [43, 57, 65, 13], [43, 92, 16, 70], [6, 46, 16, 67], [71, 22, 86, 20], [18, 72, 72, 96], [44, 87, 85, 45], [83, 13, 25, 10], [66, 6, 25, 26], [60, 6, 15, 73], [38, 41, 31, 7], [49, 29, 76, 10], [96, 85, 13, 52], [91, 23, 74, 14], [61, 24, 79, 40], [3, 43, 17, 69], [93, 91, 81, 69], [42, 55, 91, 16], [93, 84, 65, 16], [76, 23, 85, 20], [89, 96, 14, 14], [13, 95, 50, 53], [21, 58, 18, 46], [1, 86, 68, 75], [98, 40, 84, 9], [98, 10, 2, 22], [84, 17, 89, 92], [90, 42, 78, 59], [1, 27, 36, 83], [88, 77, 96, 23], [27, 36, 56, 99], [25, 78, 59, 88], [76, 93, 35, 40], [39, 28, 6, 11], [64, 55, 77, 82], [81, 16, 4, 82], [2, 25, 56, 68], [36, 58, 14, 6], [1, 15, 60, 56], [89, 36, 74, 79], [46, 66, 61, 93], [4, 47, 38, 83], [39, 28, 96, 83], [85, 23, 80, 61], [62, 40, 7, 37], [39, 24, 50, 21], [59, 12, 51, 30], [75, 88, 64, 67], [77, 44, 89, 36], [65, 73, 67, 18], [34, 57, 49, 65], [66, 67, 21, 43], [74, 61, 6, 95], [12, 89, 12, 84], [57, 78, 49, 62], [38, 86, 83, 10], [44, 99, 23, 1], [10, 49, 84, 99], [6, 73, 75, 91], [63, 58, 78, 32], [5, 18, 91, 80], [84, 70, 58, 12], [79, 84, 79, 43], [5, 47, 21, 42], [77, 46, 48, 39], [93, 7, 61, 79], [43, 18, 77, 66], [58, 71, 14, 79], [58, 74, 47, 21], [16, 10, 58, 99], [36, 79, 95, 76], [86, 52, 3, 82], [94, 24, 38, 1], [90, 88, 5, 69], [88, 25, 89, 34], [7, 6, 35, 69], [60, 23, 36, 38], [55, 78, 91, 19], [26, 35, 93, 58], [15, 21, 17, 19], [98, 24, 91, 54], [56, 45, 97, 21], [34, 2, 1, 3], [77, 61, 61, 85], [28, 12, 51, 73], [16, 63, 49, 61], [59, 94, 51, 59], [34, 79, 26, 76], [80, 69, 10, 57], [46, 81, 83, 51], [47, 35, 92, 34], [98, 69, 6, 80], [81, 52, 89, 11], [80, 69, 42, 80], [4, 76, 46, 80], [59, 64, 71, 93], [43, 14, 76, 62], [32, 17, 54, 71], [79, 89, 20, 11], [32, 74, 6, 10], [9, 49, 68, 26], [64, 91, 86, 30], [37, 89, 74, 25], [23, 20, 33, 71], [72, 85, 33, 51], [18, 31, 81, 76], [19, 25, 36, 93], [99, 10, 88, 87], [93, 18, 86, 60], [10, 65, 36, 23], [17, 64, 24, 47], [43, 39, 71, 79], [12, 34, 79, 60], [21, 38, 3, 62], [83, 84, 67, 63], [52, 95, 10, 15], [86, 23, 46, 94], [49, 99, 50, 19], [96, 56, 80, 89], [63, 72, 95, 19], [78, 61, 82, 27], [78, 29, 39, 29], [38, 28, 91, 56], [61, 3, 59, 46], [64, 50, 11, 52], [47, 50, 66, 75], [73, 25, 62, 72], [50, 49, 48, 96], [50, 93, 95, 26], [41, 74, 16, 72], [32, 3, 36, 6], [16, 31, 77, 19], [10, 26, 99, 89], [10, 98, 74, 89], [13, 57, 99, 39], [8, 82, 63, 81], [34, 45, 89, 81], [26, 81, 11, 35], [45, 7, 66, 60], [38, 48, 36, 47], [89, 85, 22, 61], [90, 85, 15, 45], [27, 90, 98, 46], [68, 65, 59, 11], [3, 60, 43, 37], [49, 63, 84, 7], [65, 52, 8, 28], [36, 19, 40, 7], [19, 76, 37, 73], [40, 99, 60, 90], [60, 40, 58, 47], [25, 64, 4, 65], [91, 53, 64, 23], [92, 57, 25, 28], [80, 59, 96, 83], [14, 2, 37, 62], [76, 92, 45, 19], [77, 45, 81, 92], [89, 94, 2, 42], [32, 29, 85, 68], [32, 7, 33, 11], [56, 17, 37, 78], [86, 47, 79, 91], [60, 62, 59, 77], [46, 28, 12, 42], [9, 23, 46, 99], [8, 29, 68, 76], [79, 17, 79, 26], [45, 17, 92, 87], [45, 57, 77, 68], [69, 76, 83, 80], [90, 89, 72, 57], [54, 86, 22, 99], [79, 54, 83, 44], [42, 83, 99, 25], [75, 96, 68, 50], [98, 57, 94, 42], [92, 19, 72, 19], [37, 38, 9, 70], [19, 43, 58, 83], [85, 62, 25, 51], [95, 34, 58, 47], [24, 2, 76, 10], [70, 72, 57, 77], [40, 21, 74, 91], [73, 78, 76, 59], [89, 78, 35, 86], [66, 25, 84, 60]]
    df=pd.read_csv("data/iris/iris.data",sep=',',header=None,names=['sepal_length','sepal_width','petal_length','petal_width','class'])
    column_names = ['sepal_length','sepal_width','petal_length','petal_width','class']
    # weights=[[1, 1, 1, 1], [36, 24, 31, 86]]
    pol=[3.6,2.4,5.9,2.4]
    p=[2,2,2,2]
    dfNew = pd.DataFrame(columns = column_names)
    dfVersi=df.loc[df['class']=='Iris-versicolor']
    dfSetosa=df.loc[df['class']=='Iris-setosa']
    dfVirginica=df.loc[df['class']=='Iris-virginica']
    for i in range(0,50,5):
        
        dfNew = pd.concat([dfNew, dfVersi.iloc[i:i+5]]).reset_index(drop=True)
        dfNew = pd.concat([dfNew, dfSetosa.iloc[i:i+5]]).reset_index(drop=True)
        dfNew = pd.concat([dfNew, dfVirginica.iloc[i:i+5]]).reset_index(drop=True)
    # print(dfNew.head(30))
    # print(dfNew.tail(30))

    arrayComp=[]
    allValues={}
    # print(dfNew.head(20))
    for w in weights:
        arrayAcc=[]
        arrayComp=[]
        arrayTimePreds=[]
        # if l==0:
        #     w=[1,1,1,1]
        # else:
        #     w= list(np.random.randint(1,100,4))
        for i in range(10):
            dfTest  = dfNew.truncate(before=i*15, after=(i+1)*15 - 1)
            dfTrain = dfNew.drop(labels=range(i*15, (i+1)*15), axis=0)

            m= dfTrain.shape[0]
            data = dfTrain.values
            dataOut= dfTrain['class']
            col=len(w)
            matrice_outcome = np.empty((m, m))
            matrice_similarity = np.empty((m, m))

            for i in range(m):
                for j in range(m):
                    if i != j:
                        matrice_similarity[i][j] =  calcSim(data,i,j,pol,p,w)
                        matrice_outcome[i][j] =  getOut(data[i][col],data[j][col])
                    else:
                        matrice_similarity[i][j] = 1
                        matrice_outcome[i][j] = 1           
            c=compl(dfTrain,w)
            startTime = time.process_time()
            acc=predictAcc(dfTest,dfTrain,c,w)
            timePred=time.process_time() - startTime
            arrayAcc.append(acc)
            arrayComp.append(c)
            arrayTimePreds.append(timePred)
        
        accuracies.append(statistics.mean(arrayAcc))
        complexities.append(statistics.mean(arrayComp))
        accuracySd.append(statistics.stdev(arrayAcc))
        complexitiesSd.append(statistics.stdev(arrayComp))
        timePreds.append(statistics.mean(arrayTimePreds))
        timePredsSd.append(statistics.mean(arrayTimePreds))



    dataReturn={
        'accuracies':accuracies,
        'complexities':complexities,
        'accuracySd':accuracySd,
        'complexitiesSd':complexitiesSd,
        'timePreds':timePreds,
        'timePredsSd':timePredsSd
    }
# print(f'time of whole code: {time.process_time() - start}')
# f = open("myfile2.txt", "x")

print(dataReturn)
# print(goodWeights)

# plt.xlim(0.5,1)

# plt.errorbar(x, y, yerr=ex, xerr=e, fmt='o')
# plt.show()
