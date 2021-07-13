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
    dataOut = train['diagnosis'].values

    simSorted=np.argsort(simArr)
    sum2=0
    sum4=0
    simSortedDesc=simSorted[::-1][:n]
    for i in range(n):
        # print(f'{i} nearest neighbor is {dataOut[simSortedDesc[i]]}')
        if dataOut[simSortedDesc[i]]==2:
            sum2+=1
        elif dataOut[simSortedDesc[i]]==4:
            sum4+=1
    maxClass=max(sum2,sum4)
    if maxClass==sum2:
        return 2
    elif maxClass==sum4:
        return 4
def calculateAug(df,minDiff,simArr,t):
    col=df.shape[1] - 1
    outArr =[]
    data = df.values
    dataOut = df['diagnosis'].values
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

        originalClass=row['diagnosis']
        for r in potential_outcomes:
            rowCopy=row
            rowCopy['diagnosis']=r
            t=rowCopy.values


            simArr = []

            data = train.values


            for i in range(m):        
                simArr.append(calcOneSim(t,data[i],pol,p,w))
            # classOpt=knn(simArr,train,5)
            simArr.append(1)

            aug=calculateAug(train,minDiff,simArr,t)

            if aug<=minDiff:
                minDiff=aug
                classOpt=r
          
        if originalClass==classOpt:
            total+=1

    return total/test.shape[0]

if __name__ == '__main__':
    start = time.process_time()
 
    weights, accuracies, complexities,  accuracySd,complexitiesSd,timePreds,timePredsSd=([] for i in range(7))
    
    df=pd.read_csv("data/breast-w/breast-cancer-wisconsin.data",sep=',',header=None,names=['id','clump_thickness','uniformity_cell_size','uniformity_cell_shape','marginal_adhesion','single_epithelial_cell_size','bare_nuclei','bland_chromatin','normal_nucleoli','mitoses','diagnosis'])
  
    pol=[3.6,2.4,5.9,2.4]
    p=[2,2,2,2]

    df = df.sample(frac=1).reset_index(drop=True)

    dfNew=df.drop(columns=['id'])
    lengthDf=dfNew.shape[1] - 1
    # p=np.full((1, lengthDf), 2)
    p=np.empty(lengthDf)
    pol=np.empty(lengthDf)
    i=0
    for i in range(0,lengthDf):

        maxVal = dfNew[dfNew.columns[i]].astype(str).map(len).max()
        pol[i]=maxVal

    p.fill(2)

    arrayComp=[]
    allValues={}
    # print(dfNew.head(20))
    for l in range(200):
        arrayAcc=[]
        arrayComp=[]
        arrayTimePreds=[]
        if l==0:
            w=[1,1,1,1]
        else:
            w= list(np.random.randint(1,100,4))
        for i in range(10):
            dfTest  = dfNew.truncate(before=i*69, after=(i+1)*69 - 1)
            dfTrain = dfNew.drop(labels=range(i*69, (i+1)*69), axis=0)

            m= dfTrain.shape[0]
            data = dfTrain.values
            dataOut= dfTrain['diagnosis']
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
        
        weights.append(w)
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
print(weights) 
print(dataReturn)
# print(goodWeights)

# plt.xlim(0.5,1)

# plt.errorbar(x, y, yerr=ex, xerr=e, fmt='o')
# plt.show()
