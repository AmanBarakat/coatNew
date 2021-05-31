import math

class Variation:
    def __init__(self):
        pass
    def apply(self,x,y):
        pass

class One(Variation):
    def apply(self,x,y):
        return 1

class Indicator(Variation):
    def apply(self,x,y):
        return 0 if x == y and x == 1 else 1

class Equal(Variation):
    def apply(self,x,y):
        return 1 if x == y else 0

class NEqual(Variation):
    def apply(self,x,y):
        return 0 if x == y else 1

class GreaterOrEqual(Variation):
    def apply(self,x,y):
        return 0 if x >= y else 1

class Minus(Variation):
    def apply(self,x,y):
        return y - x

class AbsDiff(Variation):
    def apply(self,x,y):
        return abs(y - x)

class Id(Variation):
    def apply(self,x,y):
        return (x,y)

class Step(Variation):
    def __init__(self,step):
        self.step = step

    def apply(self,x,y):
        return 1 if abs(y-x) < self.step else 0

class Polynomial(Variation):

    def __init__(self,power,value_range):
        self.power = power
        self.value_range = value_range

    def apply(self,x,y):
        return pow(abs(self.value_range-abs(y-x)),self.power)/pow(self.value_range,self.power)

class WeightedSum(Variation):
    def __init__(self,weights,ndarray,scales):
        self.w = weights
        self.scales = scales
    def apply(self,si,sj):
        r = 0.0
        for k in range(len(self.scales)):
            r+=self.w[k]*self.scales[k].apply(si[k],sj[k])
        return r/sum(self.w)