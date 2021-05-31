
import numpy as np
from scipy.spatial.distance import cosine
import pandas as pd
import math
import sys
import matplotlib.pyplot as plt
import itertools

def maxCpm(n):
    inv=0
    # x= np.random.randint(1,41,n)
    x= np.ones(n)

    z= np.random.randint(0,200,n)
    df=pd.DataFrame(data={'x': x, 'z':z})
    outcome_df = df['z']
    data = df['x']
    dataOut= outcome_df
    mo=n
    m=mo
    matrice_outcome = np.empty((m, m))
    matrice_similarity = np.empty((m, m))

    
    for i in range(m):
        for j in range(m):
            if i != j:
                matrice_similarity[i][j] =  1/(1+abs(data[i]- data[j]))
                matrice_outcome[i][j] = 1/(1+abs(dataOut[i]- dataOut[j]))
            else:
                matrice_similarity[i][j] = 1
                matrice_outcome[i][j] = 1

    tst=0
          
    for x in range(len(matrice_outcome)):
        for a, b in itertools.combinations_with_replacement(enumerate(matrice_outcome[x]), 2):
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
                    # print(f'an inversion in  {x} {small} {big}')
                    inv+=1
    return inv
def total(n):
    n_fac = math.factorial(n)
    k_fac = math.factorial(2)
    n_minus_k_fac = math.factorial(n - 2)
    return n*n_fac/(k_fac*n_minus_k_fac)

if __name__ == '__main__':
    max = 0
    n=6
    for i in range(1000):
        t=maxCpm(n)
        if max<=t:
            max=t
    print(max)
    print(total(n))