
import numpy as np
from scipy.spatial.distance import cosine
import pandas as pd
import math
import itertools
import sys
from sklearn import preprocessing
from copy import deepcopy

import time

def poly(p,r,y,x):
    return pow(abs(r-abs(y-x)),p)/pow(r,p)

def calcSim(data,y,x,pol,p,w):
    s=0
    for i in range(len(w)) :
        s+=poly(p[i], pol[i], data[x][i],data[y][i])*w[i]
    return s/sum(w)
def calcSimNormalized(data,y,x,w):
    s=0
    for i in range(len(w)) :
        s+=(1/(1+abs(data[x][i] - data[y][i])))*w[i]
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
        self.isClassification=isClassification
        self.outcome=outcomeClass
        self.inv = 0
        self.n_col = self.df.shape[1] - 1
        self.pol_ranges=pol
        self.pow_arr=powerArr

    def loadData(self):
        df = pd.read_csv(f'data/{self.name}/{self.name}.data',sep=',',names=self.att_names)
        return df
     
    def normalize(self,df):
        result = df.copy()
        for feature_name in df.columns:
            max_value = df[feature_name].max()
            min_value = df[feature_name].min()
            if max_value != min_value:
                result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
            else:
                result[feature_name]=0
        return result

    def fillSimMatrices(self,weights):

        self.matrice_outcome = np.empty((self.df.shape[0], self.df.shape[0]))
        self.matrice_similarity = np.empty((self.df.shape[0], self.df.shape[0]))
        if len(weights)>0:
            w=weights
        else:
            w= np.ones(self.n_col)
        # w=[0.625,1]
        dfData=self.df.iloc[:,:-1]
        dfDataNorm = (dfData-dfData.min())/(dfData.max()-dfData.min())
        data = dfDataNorm.values

        dataOut = self.df[self.outcome].values
        for i in range(self.df.shape[0]):
            for j in range(self.df.shape[0]):
                if i != j:
                    self.matrice_similarity[i][j] = calcSimNormalized(data,i,j,w)
                    if self.isClassification:
                        self.matrice_outcome[i][j] =  getOut(dataOut[i],dataOut[j])
                    else:
                        self.matrice_outcome[i][j] =  1/(1+abs(dataOut[i]-dataOut[j]))
                else:
                    self.matrice_similarity[i][j] = 1
                    self.matrice_outcome[i][j] = 1

    def complexity(self,printTime=True,weights=[]):
        inversions=0
        startMatrice = time.process_time()
        self.fillSimMatrices(weights)
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
                        inversions+=1
        self.inv=inversions
        if printTime:
            print(f'time of  coat calcul: {time.process_time() - startCoat}')

        return inversions

def getDataset(name):
    if name == 'autos':
        ds=Dataset('autos',[240,265,40282],[2,2,2],["horsepower","engine_size","price"],'price',False)
    elif name == 'user':
        ds=Dataset("user",[1,1,1,1,1],[2,2,2,2,2],['STG','SCG','STR','LPR','PEG','UNS'],'UNS',True)
    elif name == 'iris':
        ds=Dataset('iris', [3.6,2.4,5.9,2.4],[2,2,2,2],['sepal_length','sepal_width','petal_length','petal_width','class'],'class',True)
    elif name == 'cars':
        ds=Dataset('cars', [3,3,5,6,2,2],[2,2,2,2,2,2],['buying','maint','doors','persons','lug_boot','safety','class'],'class',True)
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


    elif name=='predict':
        if len(sys.argv) > 2:
            ds=getDataset(sys.argv[2])
            potential_outcomes = list(range(6905,10005,100))
            compOpt=0
            priceOpt=0
            c=ds.complexity(False,[9,2])
            print(c)
            minDiff=c
            for r in potential_outcomes:
                t = {'horsepower':100,'engine_size':109,'price':r}
                bs = deepcopy(ds)
                bs.df=bs.df.append(t, ignore_index=True)
                v = bs.complexity(False,[9,2])
                if v-c<=minDiff:
                    minDiff=v-c
                    compOpt=v
                    priceOpt=r
                print(f'price={r}, complexity={v}')

            print(f'complexité optimale {compOpt} for price {priceOpt}')
    elif name=='weights':
        if len(sys.argv) > 2:
            ds=getDataset(sys.argv[2])
            n_col=ds.n_col
            a_list=range(0,11,1)
            combinations_object = itertools.permutations(a_list, n_col)

            combinations_list = list(combinations_object)
            comp=ds.complexity(False)
            idealComp=np.ones(n_col)
            for com in combinations_list:
                weightComplexity = ds.complexity(False,com)
                if weightComplexity<=comp:
                    comp=weightComplexity
                    idealComp=com
                print(f'comp={weightComplexity}, array of weights={com}')
            print('=========== done ============')
            print(f'ideal comp={comp}, array of weights={idealComp}')
    else:
        ds=getDataset(name)
        c=ds.complexity()
        print(c)