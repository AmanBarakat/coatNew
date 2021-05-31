import pandas as pd
import numpy as np
import math
from simmatrix import simmatrix
from variation import Variation,AbsDiff,Equal,Polynomial,WeightedSum
from recoat import complexity,predict,concept,compatibility,delta_gamma
from evaluation import MultiClassConfusionMatrix
import matplotlib.pyplot as plt
import time
from random import shuffle

def iris():
    total_start=time.time()
    start=time.time()
    df=pd.read_csv("data/iris/iris.data",sep=',',header=None,names=['sepal_length','sepal_width','petal_length','petal_width','class'])
    end=time.time()
    print(f'load dataset in {end-start:.5f}s')
    start=time.time()
    s = simmatrix(df[['sepal_length','sepal_width','petal_length','petal_width']],WeightedSum([1,1,1,1],df,[Polynomial(2,3.6),Polynomial(2,2.4),Polynomial(2,5.9),Polynomial(2,2.4)]))
    o = simmatrix(df['class'],Equal())
    end=time.time()
    print(f'sim matrices in {end-start:.5f}s')
    start=time.time()
    c = complexity(s,o)
    end=time.time()
    total_end=time.time()
    print(f'complexity in {end-start:.5f}s')
    print(c)
    print(f'total time: {total_end-total_start:.5f}s')

def predict_iris():
    df=pd.read_csv("data/iris/iris.data",sep=',',header=None,names=['sepal_length','sepal_width','petal_length','petal_width','class'])
    

def x():
    df=pd.DataFrame(data={'x': [10, 8, 7, 9], 'A':[1,1,1,1]})
    s = simmatrix(df['A'],Equal())
    o = simmatrix(df['x'],Polynomial(2,15))
    xlist = list(range(15))
    clist = []
    for x in xlist:
        c=complexity(s.add([1]),o.add([x]))
        print("x=%d : %d " % (x,c))
        clist.append(c)
    p=predict(s,o,[1],xlist)
    print(f'predict={p}')
    plt.plot(xlist,clist,color='red')
    plt.show()

def nn():
    n=100
    nb_ones=list(range(n))
    c_results=[]
    start=time.time()
    for k in nb_ones:
        x_list=[0]*(n-k)+[1]*k
        shuffle(x_list)
        df=pd.DataFrame(data={'x': x_list, 'A':[1]*n})
        s = simmatrix(df['A'],Equal())
        o = simmatrix(df['x'],Equal())
        c_results.append(complexity(s,o))
    end=time.time()
    print(f'{n} complexity runs in {end-start:.5f}s')
    # one_results=[]
    # zero_results=[]
    # start=time.time()
    # for k in nb_ones:
    #     x_list=[0]*(n-k)+[1]*k
    #     shuffle(x_list)
    #     df=pd.DataFrame(data={'x': x_list, 'A':[1]*n})
    #     s = simmatrix(df['A'],Equal())
    #     o = simmatrix(df['x'],Equal())
    #     one_results.append(predict(s,o,[1],[1],return_complexity=True)[0])
    #     zero_results.append(predict(s,o,[1],[0],return_complexity=True)[0])
    # end=time.time()
    # print(f'{n} predict runs in {end-start:.5f}s')
    # plt.plot(nb_ones,zero_results,color='#55557fff')
    # plt.plot(nb_ones,one_results,color='red')
    plt.plot(nb_ones,c_results,color='#55557fff')
    plt.show()

def price():
    df = pd.read_csv("data/apartments-num.csv",sep=';')
    s = simmatrix(df[['nb_rooms','area']],WeightedSum([1,1],df,[Polynomial(2,6),Equal()]))
    o = simmatrix(df['price'],Polynomial(2,800))
    potential_outcomes = list(range(400,1200,1))
    print(predict(s,o,[[2,1]],potential_outcomes))

def nb_rooms():
    df = pd.read_csv("data/apartments-num.csv",sep=';')
    s = simmatrix(df[['nb_rooms','area']],WeightedSum([1,1],df,[Polynomial(2,6),Equal()]))
    o = simmatrix(df['price'],Polynomial(2,800))
    potential_outcomes = list(range(400,1200,1))
    results = []
    for r in potential_outcomes:
        c = complexity(s.add([[2,1]]),o.add([r]))
        print(f'r={r} c={c}')
        results.append(c)
    plt.figure(figsize=(10,6))
    plt.rc('axes', labelsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel(r'Potential price for $t$')
    plt.ylabel(r'Complexity indicator')
    #plt.grid(True)
    plt.xlim(400,1200)
    plt.ylim(0,max(results)+1)
    line, = plt.plot(potential_outcomes,results,color='#55557fff')
    line.set_linewidth(2)
    plt.axhline(y=4, color='#55557fff', linestyle='dashed')
    plt.annotate(text='', xy=(450,0), xytext=(450,4), arrowprops=dict(arrowstyle='<->'))
    plt.annotate(text=r'Complexity of $CB$ without $t$', xy=(470,2),fontsize=14)
    plt.tight_layout()
    plt.show()

class ES(Variation):
    def apply(self,a,b):
        return 1/(1+math.sqrt(AbsDiff().apply(a,b)**2))
    
def AB():
    df=pd.DataFrame(data={'x': [1, 2], 'A':[0,1]})
    s = simmatrix(df['x'],ES())
    o = simmatrix(df['A'],Equal())
    print(complexity(s,o))
    print(complexity(s.add([0]),o.add([1])))
    print(complexity(s.add([0]),o.add([0])))
    print(predict(s,o,[0],[0,1],return_complexity=True))

def AB_given_C():
    df=pd.DataFrame(data={'x': [1, 2], 'A':[0,1]})
    s = simmatrix(df['x'],ES())
    o = simmatrix(df['A'],Equal())
    c = concept(Polynomial(2,1),0,Polynomial(2,1),0)
    n=len(s)+1
    d=len([0,1])
    x_list=[e/10 for e in list(range(-10,20))]
    p=np.zeros((len(x_list),d))
    for k in range(len(x_list)):
        x=x_list[k]
        khi=np.zeros((n,d))
        gamma=np.zeros((n,d))
        pi=np.zeros((n,d))
        for i in range(n):
            for rt in [0,1]:
                khi[i][rt]=compatibility(s,o,[x],[rt],i,c)
                gamma[i][rt]=delta_gamma(s,o,[x],[rt],i)
            g=np.sum(gamma[i])
            for j in range(d):
                pi[i][j]=1-gamma[i][j]/n
        for j in range(d):
            for i in range(n):
                p[k][j]+=khi[i][j]*pi[i][j]
    plt.plot(x_list,list(p[:,1]),color='#55557fff')
    plt.plot(x_list,list(p[:,0]),color='red')
    plt.show()

AB_given_C()
