from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

class MyRegression:
    def __init__(self, data):
        self.data = data
        
    def createTrainTestData(self, X, y, train_size=0.75, random_state=100):
        X = self.data[X]
        y = self.data[y]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, train_size=0.75)
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }
        
    def simpleLinearRegression(self, X_train, y_train):
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        return model