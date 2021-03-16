from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

class MySimpleLinearRegression:
    def __init__(self, data):
        self.data = data
        
    def createTrainTestData(self, X, y, train_size=0.75, random_state=100):
        self.X = self.data[[X]]
        self.y = self.data[y]
        self.train = {}
        self.test = {}

        self.train['X'], self.test['X'], self.train['y'], self.test['y'] = train_test_split(
            self.X, self.y, random_state=random_state, train_size=train_size)
        
    def initModel(self):
        self.model = LinearRegression()
        self.model.fit(self.train['X'], self.train['y'])
        
    def getFormula(self):
        return {
            'intercept': self.model.intercept_,
            'coef': self.model.coef_[0]
        }
        
    def predict(self, X=None):
        if X is None:
            X = self.test['X']
            
        return self.model.predict(X)
    
    def r2(self, option='all'):
        if option == 'all':
            return self.model.score(self.X, self.y)
        elif option == 'train':
            return self.model.score(self.train['X'], self.train['y'])
        else:
            return self.model.score(self.test['X'], self.test['y'])
        
    def leastSquares(self, y_predict, option='mse'):
        if option == 'mse':
            return mean_squared_error(y_predict, self.test['y'])
        elif option == 'mae':
            return mean_absolute_error(y_predict, self.test['y'])
            
            
class MyMultipleLinearRegression:
    def __init__(self, data):
        self.data = data
        
    def createTrainTestData(self, X, y, train_size=0.75, random_state=100):
        self.X = self.data[X]
        self.y = self.data[y]
        self.train = {}
        self.test = {}

        self.train['X'], self.test['X'], self.train['y'], self.test['y'] = train_test_split(
            self.X, self.y, random_state=random_state, train_size=train_size)
        
    def initModel(self):
        self.model = LinearRegression()
        self.model.fit(self.train['X'], self.train['y'])
        
    def predict(self, X=None):
        if X is None:
            X = self.test['X']
            
        return self.model.predict(X)
    
    def getFormula(self):
        return {
            'intercept': self.model.intercept_,
            'coef': self.model.coef_
        }
        
    def r2(self, option='all'):
        if option == 'all':
            return self.model.score(self.X, self.y)
        elif option == 'train':
            return self.model.score(self.train['X'], self.train['y'])
        else:
            return self.model.score(self.test['X'], self.test['y'])
        
    def leastSquares(self, y_predict, option='mse'):
        if option == 'mse':
            return mean_squared_error(y_predict, self.test['y'])
        elif option == 'mae':
            return mean_absolute_error(y_predict, self.test['y'])