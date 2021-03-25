from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd

class CRandomForest:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def prepareData(self, train_size=None, random_state=None):
        self.train, self.test = {}, {}
        self.train['X'], self.test['X'], self.train['y'], self.test['y'] = train_test_split(
            self.X, self.y, random_state=random_state, train_size=train_size)


    def initModel(self, option='classification', no_trees=100):
        if option == 'classification':
            self.model = RandomForestClassifier()
        elif option == 'regression':
            self.model = RandomForestRegressor()
        self.model.fit(self.train['X'], self.train['y'])
        
    def predict(self, X=None):
        if X is None:
            X = self.test['X']

        return self.model.predict(X)
    

    def accuracy(self, X=None):
        return metrics.accuracy_score(self.test['y'], self.predict(X))