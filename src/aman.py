
import numpy as np
from scipy.spatial.distance import cosine
import pandas as pd
import math
import itertools
import sys

from copy import deepcopy


import time

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

class Dataset(object):
    def __init__(self,name,pol,powerArr,att_names,outcomeClass,isClassification,potential_outcomes):
        self.name = name
        self.att_names=att_names
        self.df=self.loadData()
        self.isClassification=isClassification
        self.outcome=outcomeClass
        self.inv = 0
        self.potential_outcomes=potential_outcomes
        self.n_col = self.df.shape[1] - 1
        self.pol_ranges=pol
        self.pow_arr=powerArr

    def loadData(self):
        df = pd.read_csv(f'data/{self.name}/{self.name}.data',sep=',',names=self.att_names)
        return df
    def appendToData(self,t):
        self.df=self.df.append(t, ignore_index=True)
    
    def dropFromData(self,t):
        self.df.drop(self.df.tail(1).index,inplace=True) 

 
    def fillSimMatrices(self):
        self.matrice_outcome = np.empty((self.df.shape[0], self.df.shape[0]))
        self.matrice_similarity = np.empty((self.df.shape[0], self.df.shape[0]))
        # w= np.ones(self.n_col)
        w=[0.625,1]

        data = self.df.values

        dataOut = self.df[self.outcome].values
        for i in range(self.df.shape[0]):
            for j in range(self.df.shape[0]):
                if i != j:
                    self.matrice_similarity[i][j] =  calcSim(data,i,j,self.pol_ranges,self.pow_arr,w)
                    if self.isClassification:
                        self.matrice_outcome[i][j] =  getOut(dataOut[i],dataOut[j])
                    else:
                        self.matrice_outcome[i][j] =  poly(self.pow_arr[self.n_col],self.pol_ranges[self.n_col],dataOut[i],dataOut[j])
                else:
                    self.matrice_similarity[i][j] = 1
                    self.matrice_outcome[i][j] = 1
    def complexity(self,printTime=True):
        inversions=0
        startMatrice = time.process_time()
        self.fillSimMatrices()
        if printTime:
            print(f'time of matrices calcul: {time.process_time() - startMatrice}')

        startCoat = time.process_time()
        for x in range(self.df.shape[0]):
            for a, b in itertools.combinations(enumerate(self.matrice_outcome[x]), 2):
                if a[1] == b[1]:
                    continue
                else:
                    if a[1]>b[1]:
                        big=a[0]
                        small=b[0]
                    else:
                        big=b[0]
                        small=a[0]
                    if self.matrice_similarity[x][small] - self.matrice_similarity[x][big]>=0:
                        # print(f'an inversion in  {x} {small} {big}')
                        inversions+=1
        self.inv=inversions
        if printTime:
            print(f'time of  coat calcul: {time.process_time() - startCoat}')

        return inversions
    def calculateAug(self,minDiff,w):
        m=self.df.shape[0]-1
        outArr =[]
        simArr = []
        data = self.df.values
        dataOut = self.df[self.outcome].values

        for i in range(m):
            # print(i)
            simArr.append(calcSim(data,m,i,self.pol_ranges,self.pow_arr,w))
            if self.isClassification:
                outArr.append(getOut(dataOut[m],dataOut[i]))
            else:
                outArr.append(poly(self.pow_arr[self.n_col],self.pol_ranges[self.n_col],dataOut[m],dataOut[i]))
        simArr.append(1)
        outArr.append(1)
        newInv=0
        while newInv <= minDiff:
            for x in range(m):
                simVal =  calcSim(data,x,m,self.pol_ranges,self.pow_arr,w)
                outVal =  poly(2,40282,dataOut[x],dataOut[m])

                for y in range(m):
                    if self.matrice_outcome[x][y] == outVal:
                        continue
                    else:
                        if  self.matrice_outcome[x][y] > outVal:
                            if self.matrice_similarity[x][y] <= simVal: 
                                newInv+=1
                        elif self.matrice_similarity[x][y] >= simVal:
                            newInv+=1
            for a, b in itertools.combinations(enumerate(outArr), 2):
                    # print(f'a is {a}, b is {b}')
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
def getDataset(name):
    if name == 'autos':
        ds=Dataset('autos',[240,265,40282],[2,2,2],["horsepower","engine_size","price"],'price',False,list(range(8000,9000,100)))
    elif name == 'user':
        ds=Dataset("user",[1,1,1,1,1],[2,2,2,2,2],['STG','SCG','STR','LPR','PEG','UNS'],'UNS',True)
    elif name == 'iris':
        ds=Dataset('iris', [3.6,2.4,5.9,2.4],[2,2,2,2],['sepal_length','sepal_width','petal_length','petal_width','class'],'class',True,['Iris-versicolor','Iris-virginica','Iris-setosa'])
    elif name == 'cars':
        ds=Dataset('cars', [3,3,5,6,2,2],[2,2,2,2,2,2],['buying','maint','doors','persons','lug_boot','safety','class'],'class',True)
    return ds
if __name__ == '__main__':
    print(list(range(8000,9000,100)))
    name = sys.argv[1]
    if name=='tests':
        if len(sys.argv) > 2:
            dataNames=[sys.argv[2]]
        else:
            dataNames=['autos','user','iris']
        for name in dataNames:
            ds=getDataset(name)
            durations=[]
            for i in range(10):
                startTime = time.process_time()
                c=ds.complexity(False)
                durations.append(time.process_time() - startTime)
            meanTime=np.mean(durations)
            stdTime=np.std(durations)
            print(f'{name} : mean = {meanTime} , std = {stdTime}')


    elif name=='predict':
        if len(sys.argv) > 2:
            ds=getDataset(sys.argv[2])
            compOpt=0
            classOpt=''
            c=ds.complexity(False)
            # print(c)
            w=[0.625,1]

            m=ds.df.shape[0]
            minDiff=(m*m)+(2*m)
            for r in ds.potential_outcomes:
                t = {'horsepower':116,'engine_size':110,'price':r}
                ds.appendToData(t)
                print(len(ds.matrice_outcome))
                aug=ds.calculateAug(minDiff,w)
                if aug<=minDiff:
                    minDiff=aug
                    priceOpt=r
                # print(f'price={r}, complexity={v}')
                ds.dropFromData(t)
            print(f'complexité optimale {minDiff} for class {priceOpt}')

    else:
        ds=getDataset(name)

        c=ds.complexity()
        print(c)