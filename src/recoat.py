import numpy as np
import pandas as pd
import math

class ReCoAT(object):
    """(Re)fined measure of the (Co)mplexity of a dataset for (A)nalogical (T)ransfer."""
    def __init__(self,s,o):
        self.s = s
        self.o = o
        self.n = s.shape[0]

    def rank(self,u):
        inv=np.unique(u,return_inverse=True)[1]
        ind,pos=np.unique(np.sort(inv),return_index=True)
        return np.array([self.n-pos[list(ind).index(e)]-1 for e in inv])

    """
    Compute dataset complexity.
    """
    def gamma(self,i):
        
        si = self.s[i]
        oi = self.o[i]
        
        # ranks of o and s (with possible ties)
        ranks_o = self.rank(oi)
        inv_ranks_o = [self.n-1-e for e in ranks_o]
        ranks_s = self.rank(si)

        # ranks of s,o (= according to s in decreasing order and then o in increasing order)   
        ranks_s_o = np.arange(self.n)[np.argsort(list(np.lexsort((inv_ranks_o,ranks_s))))]

        # create n x n matrix
        m=np.zeros((self.n,self.n))

        # now read the matrix and count the inversions
        g = 0.
        for k in np.argsort(np.subtract(ranks_s_o,ranks_o)):
            a=ranks_s_o[k]
            b=ranks_o[k]
            m[a][b]=1
            g+=np.sum(m[:a+1,b+1:])

        return g

    def delta_gamma(self,i):
        r=0
        last=self.n-1
        if i==last:
            r = self.gamma(last)
        else:
            si = self.s[i]
            oi = self.o[i]
            for k in range(last):
                if oi[k]<oi[last]:
                    if si[k]>=si[last]:
                        r+=1
                elif oi[k]>oi[last]:
                    if si[k]<=si[last]:
                        r+=1
        return r

    def complexity_increase(self):
        r=0
        for i in range(self.n):
            r+=self.delta_gamma(i)
        return r
class Concept(object):
    """
    A concept, i.e., a rule A -> B where A and B are subsets of SxS
    """
    def __init__(self,varA,a,varB,b):
        self.varA=varA
        self.a=a
        self.varB=varB
        self.b=b

    def compatibility(self,simS,simR):
        #return min(1-self.varA.apply(self.a,simS),1-self.varB.apply(self.b,simR))
        return min(simS,simR)

def concept(varA,a,varB,b):
    return Concept(varA,a,varB,b)

def complexity(s,o):
    coat=ReCoAT(s,o)
    c = 0
    for i in range(coat.n):
        c+=coat.gamma(i)
    return c   

def predict(s,o,new_s,potential_outcomes,return_complexity=False):
    deltas=[]
    for v in potential_outcomes:
        coat=ReCoAT(s.add(new_s),o.add([v]))
        r = coat.complexity_increase()
        #print(f'{v} -> {r}')
        deltas.append(r)
    i = min(list(range(len(deltas))), key=(lambda k: deltas[k]))
    if return_complexity:
        return (deltas[i],potential_outcomes[i])
    else:
        return potential_outcomes[i]

def delta_gamma(s,o,t,rt,i):
    return ReCoAT(s.add(t),o.add(rt)).delta_gamma(i)        

def compatibility(s,o,t,rt,i,concept):
    return concept.compatibility(s.add(t)[i][-1],o.add(rt)[i][-1])