import numpy as np
import pandas as pd


def simmatrix(df,scale):
    if type(df) is pd.DataFrame or pd.Series:
        ndarray=df.values
    else:
        ndarray=df
    n=ndarray.shape[0]
    m=SimMatrix(ndarray,scale,shape=(n,n))
    for i in range(n):
        for j in range(i):
            m[i][j]=scale.apply(ndarray[i],ndarray[j])
    for i in range(n):
        m[i][i]=1.0
    for i in range(n):
        for j in range(i+1,n):
            m[i][j]=m[j][i]
    return m

class SimMatrix(np.ndarray):
    # https://numpy.org/doc/stable/user/basics.subclassing.html
    def __new__(subtype, ndarray, scale, shape, dtype=float, buffer=None, offset=0, strides=None, order=None, info=None):
        obj=super(SimMatrix,subtype).__new__(subtype, shape, dtype, buffer, offset, strides, order)
        obj.df=ndarray
        obj.n=ndarray.shape[0]
        obj.scale=scale
        return obj        

    def __array_finalize__(self,obj):
        if obj is None: return
        self.df=getattr(obj,'df',None)
        self.n=getattr(obj,'n',None)
        self.scale=getattr(obj,'scale',None)

    def fill_row(self,i):
        """
        Fills ith row of the matrix.
        """
        for j in range(self.n):
            self[i][j]=self.scale.apply(self.df[i],self.df[j])

    def fill_column(self,j):
        for i in range(self.n):
            self[i][j]=self.scale.apply(self.df[i],self.df[j])

    def add(self,array):
        """
        Adds a line to the original array,
        and update sim matrix accordingly.
        """
        df = np.concatenate((self.df,np.array(array)))
        new_m=SimMatrix(df,self.scale,shape=(self.n+1,self.n),buffer=np.concatenate((self,np.empty((1,self.n))),axis=0))
        new_m=SimMatrix(new_m.df,new_m.scale,shape=(new_m.n,new_m.n),buffer=np.concatenate((new_m,np.empty((new_m.n,1))),axis=1))
        new_m.fill_row(new_m.n-1)
        new_m.fill_column(new_m.n-1)
        return new_m

        


    
        
        