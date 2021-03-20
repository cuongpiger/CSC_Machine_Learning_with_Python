import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from math import sqrt, ceil

class CKNearestNeighbors:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.k_value = None
            
    def prepareData(self, X=None, y=None, train_size=0.75, random_state=100):
        self.train = {}
        self.test = {}
        
        if type(X) == list:
            self.X = self.data[X]
            self.y = self.data[y]
        elif X is not None:
            self.X = X
            self.y = y
            self.data = pd.concat([X, y.reindex(y.index)], axis=1)

        self.train['X'], self.test['X'], self.train['y'], self.test['y'] = train_test_split(
                self.X, self.y, random_state=random_state, train_size=train_size)
        
    def findKValue(self):
        error_rate = []
        
        for k in range(2, ceil(sqrt(len(self.X)/2))):
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(self.train['X'], self.train['y'])
            y_predict = model.predict(self.test['X'])
            error_rate.append((np.mean(y_predict), k))
            print('Error: {} at K: {}'.format(np.mean(y_predict), k))
            
        self.k_value = error_rate.index(min(error_rate))
        best_option = min(error_rate)
        print("Minimum error: {} at K-Value: {}".format(best_option[0], best_option[1]))
        
    def initModel(self):
        if self.k is None:
            self.findKValue()
            
        self.model = KNeighborsClassifier(n_neighbors=self.k)
        self.model.fit(self.train['X'], self.train['y'])
            
        
        
    