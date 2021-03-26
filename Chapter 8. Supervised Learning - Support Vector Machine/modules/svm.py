from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

class CSVM:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def prepareData(self, train_size=None, random_state=None):
        self.train, self.test = {}, {}
        self.train['X'], self.test['X'], self.train['y'], self.test['y'] = train_test_split(
            self.X, self.y, random_state=random_state, train_size=train_size)
        
    def initModel(self, option='classification'):
        if option == 'classification':
            self.model = svm.SVC(gamma=.001, C=100)
        elif option == 'regression':
            pass
        
        self.model.fit(self.train['X'], self.train['y'])
        
        