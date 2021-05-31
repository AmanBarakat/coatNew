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
    dataOut = df['class'].values


    for i in range(m):
        simArr.append(calcSim(data,m,i,pol,p,w))
        outArr.append(getOut(dataOut[m],dataOut[i]))
    simArr.append(1)
    outArr.append(1)
    newInv=0
    while newInv <= minDiff:
        for x in range(m):
            simVal =  calcSim(data,x,m,pol,p,w)
            outVal =  getOut(dataOut[x],dataOut[m])

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
            train=train.append(rowCopy, ignore_index=True)
            aug=calculateAug(train,minDiff)
            # print(f"pour {r} augmentation est de {aug}")
            # print(f'miDiff={minDiff}')
            if aug<=minDiff:
                minDiff=aug
                classOpt=r
                # print(f'True and now miDiff is {minDiff}')

            train.drop(train.tail(1).index,inplace=True) 

        # print(f'Classe optimale {classOpt} alors que originale est {originalClass}')

        if originalClass==classOpt:
            total+=1
    # print(f'total is {total}')
    return total/test.shape[0]

if __name__ == '__main__':
    start = time.process_time()

    x=[]
    y=[]
    e=[]
    ex=[]
    df=pd.read_csv("data/iris/iris.data",sep=',',header=None,names=['sepal_length','sepal_width','petal_length','petal_width','class'])
    column_names = ['sepal_length','sepal_width','petal_length','petal_width','class']

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
    for l in range(500):
        arrayAcc=[]
        arrayComp=[]
        if l==0:
            w=[1,1,1,1]
        else:
            w= list(np.random.randint(1,100,4))
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
            arrayAcc.append(predictAcc(dfTest,dfTrain,c,w))
            arrayComp.append(c)

        x.append(statistics.mean(arrayAcc))
        y.append(statistics.mean(arrayComp))
        e.append(statistics.stdev(arrayAcc))
        ex.append(statistics.stdev(arrayComp))

        data={
            'weights':w,
            'arrayAcc':arrayAcc,
            'meanAcc':statistics.mean(arrayAcc),
            'sdAcc':statistics.stdev(arrayAcc),
            'meanComp':statistics.mean(arrayComp),
            'sdComp':statistics.stdev(arrayComp)
        }
        allValues[l]=data
        # print(f'{l} done')

    print(allValues)
    dataReturn={
        'x':x,
        'y':y,
        'e':e,
        'ex':ex
    }
print(f'time of whole code: {time.process_time() - start}')
# f = open("myfile2.txt", "x") 
print(dataReturn)

plt.xlim(0.5,1)

plt.errorbar(x, y, yerr=ex, xerr=e, fmt='o')
plt.show()
