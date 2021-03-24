import pandas as pd
import numpy as np
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

class CNaiveBayes:
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def initModel(self, option='bernoulli'):
        if option == 'bernoulli': # dùng khi X là ma trận thưa thớt chỉ có 0 và 1
            self.model = BernoulliNB()
        elif option == 'gaussian': # dùng khi X vừa có biến phân loại và liên tục
            self.model = GaussianNB()
        elif option == 'multinomial': # dùng vs dữ liệu văn bản nh chiều
            self.model = MultinomialNB()
        
        self.model.fit(self.X, self.y)
            
    def predict(self, X=None):
        if X is None:
            X = self.test['X']

        return self.model.predict(X)