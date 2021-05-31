
import numpy as np
from scipy.spatial.distance import cosine
import pandas as pd
import math
import itertools
import sys


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
    def __init__(self,name,pol,powerArr,att_names,outcomeClass,isClassification):
        self.name = name
        self.att_names=att_names
        self.df=self.loadData()
        self.row_len=self.df.shape[0]
        self.isClassification=isClassification
        self.outcome=outcomeClass
        self.inv = 0
        self.matrice_outcome = np.empty((self.row_len, self.row_len))
        self.matrice_similarity = np.empty((self.row_len, self.row_len))
        self.n_col = self.df.shape[1] - 1
        self.pol_ranges=pol
        self.pow_arr=powerArr

    def loadData(self):
        df = pd.read_csv(f'data/{self.name}/{self.name}.data',sep=',',names=self.att_names)
        return df
     

    def fillSimMatrices(self):
        w= np.ones(self.n_col)

        data = self.df.values

        dataOut = self.df[self.outcome].values
        for i in range(self.row_len):
            for j in range(self.row_len):
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
        inv=0
        startMatrice = time.process_time()

        self.fillSimMatrices()
        if printTime:
            print(f'time of matrices calcul: {time.process_time() - startMatrice}')

        startCoat = time.process_time()
        for x in range(self.row_len):
            for y in range(self.row_len):
                for z in range(y+1,self.row_len):
                    if self.matrice_outcome[x][y]>self.matrice_outcome[x][z]:
                        if self.matrice_similarity[x][z] - self.matrice_similarity[x][y]>0:
                            inv+=1
                    elif self.matrice_outcome[x][y]<self.matrice_outcome[x][z]:
                        if self.matrice_similarity[x][z] - self.matrice_similarity[x][y]<0:
                            inv+=1
        self.inv=inv
        if printTime:
            print(f'time of  coat calcul: {time.process_time() - startCoat}')

        return inv

def getDataset(name):
    if name == 'autos':
        ds=Dataset('autos',[240,265,40282],[2,2,2],["horsepower","engine_size","price"],'price',False)
    elif name == 'user':
        ds=Dataset("user",[1,1,1,1,1],[2,2,2,2,2],['STG','SCG','STR','LPR','PEG','UNS'],'UNS',True)
    elif name == 'iris':
        ds=Dataset('iris', [3.6,2.4,5.9,2.4],[2,2,2,2],['sepal_length','sepal_width','petal_length','petal_width','class'],'class',True)
    return ds
if __name__ == '__main__':
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



    else:
        ds=getDataset(name)

        c=ds.complexity()
        print(c)