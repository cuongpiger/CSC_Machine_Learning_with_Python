import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

class CTextHandler:
    def __init__(self, data):
        self.data = data
        
    def createBagOfWords(self):
        self.handler = CountVectorizer()
        self.handler.fit(self.data)
        
        return self.handler.transform(self.data).toarray()
    
    def transform(self, data):
        return self.handler.transform(data).toarray()