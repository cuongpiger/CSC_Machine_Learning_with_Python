from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

class MySimpleLinearRegression:
    def __init__(self, data):
        self.data = data
        
    def createTrainTestData(self, X, y, train_size=0.75, random_state=100):
        X = self.data[[X]]
        y = self.data[y]
        self.X = {}
        self.y = {}

        self.X['train'], self.X['test'], self.y['train'], self.y['test'] = train_test_split(
            X, y, random_state=random_state, train_size=0.75)
        
    def initModel(self):
        self.model = LinearRegression()
        self.model.fit(self.X['train'], self.y['train'])
        
    def getFormula(self):
        return {
            'intercept': self.model.intercept_,
            'coef': self.model.coef_[0]
        }
        
    def predict(self, X=None):
        if X is None:
            X = self.X['test']
            
        return self.model.predict(X)