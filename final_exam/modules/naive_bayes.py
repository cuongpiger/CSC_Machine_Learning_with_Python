import pandas as pd
import numpy as np
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

class CNaiveBayes:
    def __init__(self, X: pd.DataFrame, y: pd.Series):
        self.X: pd.DataFrame = X
        self.y: pd.Series = y
        self.model = None

    def prepareData(self, train_size=None, random_state=None):
        self.train, self.test = {}, {}
        self.train['X'], self.test['X'], self.train['y'], self.test['y'] = train_test_split(
            self.X, self.y, random_state=random_state, train_size=train_size)

    def initModel(self, option='bernoulli'):
        if option == 'bernoulli': # dùng khi X là ma trận thưa thớt chỉ có 0 và 1
            self.model = BernoulliNB()
        elif option == 'gaussian': # dùng khi X vừa có biến phân loại và liên tục
            self.model = GaussianNB()
        elif option == 'multinomial': # dùng vs dữ liệu văn bản nh chiều
            self.model = MultinomialNB()
        
        self.model.fit(self.train['X'], self.train['y'])
            
    def predict(self, X=None):
        if X is None:
            X = self.test['X']

        return self.model.predict(X)

    def scoreR2(self):
        return pd.DataFrame({
            'Entire dataset': [self.model.score(self.X, self.y)],
            'Training data': [self.model.score(self.train['X'], self.train['y'])],
            'Test data': [self.model.score(self.test['X'], self.test['y'])]
        })

    def confusionMatrix(self):
        return confusion_matrix(self.test['y'], self.predict())

    def accuracyScore(self):
        return accuracy_score(self.test['y'], self.predict())*100.0

    def classificationReport(self):
        print(classification_report(self.test['y'], self.predict()))
