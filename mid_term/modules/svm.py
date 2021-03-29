from sklearn import model_selection
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score
from sklearn.svm import SVR
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class CSupportVectorMachine:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def prepareData(self, test_size=None, random_state=None):
        self.train, self.test = {}, {}
        self.train['X'], self.test['X'], self.train['y'], self.test['y'] = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)
            
    def initModel(self, option, kernel):
        if option == 'regression':
            self.model = SVR(kernel=kernel)
            self.model.fit(self.train['X'], self.train['y'])

    def predict(self, X=None):
        if X is None:
            X = self.test['X']
            
        return self.model.predict(X)
    
    def r2(self, X=None):
        return {
            'all': r2_score(self.y, self.predict(self.X))
        }
    

    def kfold(self, kernel, n_splits=10):
        model = SVR(kernel=kernel)
        res = model_selection.cross_val_score(model, self.X, self.y, cv=KFold(n_splits=n_splits))
        
        print('Accuracy - mean: {0:.3f}%, std: {1:.3f}%'.format(res.mean()*100., res.std()*100.))