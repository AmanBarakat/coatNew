import numpy as np

class MultiClassConfusionMatrix(object):
    def __init__(self, classes):
        super(MultiClassConfusionMatrix, self).__init__()
        self.n = len(classes)
        self.classes = classes
        self.m = np.zeros((self.n,self.n))
        self.nb_predictions = 0

    def add_prediction(self,predicted_class,real_class):
        i = self.classes.index(predicted_class)
        j = self.classes.index(real_class)
        self.m[i][j]+=1
        self.nb_predictions+=1

    def accuracy(self):
        acc = 0.0
        for k in range(self.n):
            acc+=self.m[k][k]
        return acc/self.nb_predictions