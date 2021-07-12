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
def getOut(x,y):
    if x==y:
        return 1
    else:
        return 0
def calculateAug(df,minDiff):

    outArr =[]
    simArr = []

    data = df.values
    dataOut = df['price'].values


    for i in range(m):
        simArr.append(calcSim(data,m,i,pol,p,w))
        outArr.append(poly(2,40282,dataOut[m],dataOut[i]))
    simArr.append(1)
    outArr.append(1)
    newInv=0
    while newInv <= minDiff:
        for x in range(m):
            simVal =  calcSim(data,x,m,pol,p,w)
            outVal =  poly(2,40282,dataOut[x],dataOut[m])
        
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
    potential_outcomes=list(range(5000,30000,100))
    classOpt=''
    for index, row in test.iterrows():
        minDiff = m*m*m*m
        print(f'mindiff is {minDiff}')
        originalClass=row['price']
        print(f'original price is {originalClass}')
        for r in potential_outcomes:
            
            rowCopy=row
            rowCopy['price']=r
            # print(row)
            train=train.append(rowCopy, ignore_index=True)
            aug=calculateAug(train,minDiff)
            # print(f'for r={r} augmentation was {aug}')
            if aug<=minDiff:
                minDiff=aug
                classOpt=r
            train.drop(train.tail(1).index,inplace=True)

        print(f'trouver prix optimal {classOpt}')
        if abs(originalClass - classOpt) <= 1000 :
            total+=1
    # print(f'total is {total}')
    return total/test.shape[0]

if __name__ == '__main__':
    start = time.process_time()

    x=[]
    y=[]
    e=[]
    weights=[]
    ex=[]
    column_names=["horsepower","engine_size","price"]
    df=pd.read_csv("data/autos/autos.data",sep=',',names=column_names)

    pol=[3.6,2.4,5.9,2.4]
    p=[2,2,2,2]

    arrayComp=[]
    allValues={}
    # print(dfNew.head(20))
    ww=[[1, 1], [11, 55], [33, 22], [41, 99], [57, 63], [94, 50], [19, 39], [17, 66]]
    for l in range(8):
        arrayAcc=[]
        arrayComp=[]
        # if l==0:
        #     w=[1,1,1,1]
        # else:
        #     w= list(np.random.randint(1,100,2))
        w=ww[l]
        for i in range(10):
            dfTest  = df.truncate(before=i*15, after=(i+1)*15 - 1)
            dfTrain = df.drop(labels=range(i*15, (i+1)*15), axis=0)

            m= dfTrain.shape[0]
            data = dfTrain.values
            dataOut= dfTrain['price']
            col=len(w)
            matrice_outcome = np.empty((m, m))
            matrice_similarity = np.empty((m, m))

            for i in range(m):
                for j in range(m):
                    if i != j:
                        matrice_similarity[i][j] =  calcSim(data,i,j,pol,p,w)
                        matrice_outcome[i][j] =  poly(2,40282,data[i][col],data[j][col])
                    else:
                        matrice_similarity[i][j] = 1
                        matrice_outcome[i][j] = 1
            c=compl(dfTrain,w)
            arrayAcc.append(predictAcc(dfTest,dfTrain,c,w))
            arrayComp.append(c)

        x.append(statistics.mean(arrayAcc))
        y.append(statistics.mean(arrayComp))
        e.append(statistics.stdev(arrayAcc))
        ex.append(statistics.stdev(arrayComp))


    print(allValues)
    dataReturn={
        'x':x,
        'y':y
    }
print(f'time of whole code: {time.process_time() - start}')

print(dataReturn)

plt.xlim(0.5,1)

plt.errorbar(x, y, yerr=ex, xerr=e, fmt='o')
plt.show()
