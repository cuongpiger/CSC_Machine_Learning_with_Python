from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class CLogisticRegression:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def prepareData(self, train_size=None, random_state=None):
        self.train, self.test = {}, {}
        self.train['X'], self.test['X'], self.train['y'], self.test['y'] = train_test_split(
            self.X, self.y, random_state=random_state, train_size=train_size)

    def initModel(self):
        self.model = LogisticRegression()
        self.model.fit(self.train['X'], self.train['y'])

    def predict(self, X=None):
        if X is None:
            X = self.test['X']

        return self.model.predict(X)

    def r2(self):
        return {
            'Entire dataset': self.model.score(self.X, self.y),
            'Training data': self.model.score(self.train['X'], self.train['y']),
            'Test data': self.model.score(self.test['X'], self.test['y'])
        }

    def accuracy(self, X=None):
        return accuracy_score(self.test['y'], self.predict(X))

    def confusionMatrix(self, X=None):
        return pd.DataFrame(confusion_matrix(self.test['y'], self.predict(X)))

    def getFormula(self):
        return {
            'intercept': self.model.intercept_,
            'coef': self.model.coef_
        }

    def visualTestVsPredict(self, x, y):
        fig, ax = plt.subplots(2, 2, figsize=(15, 15))
        sns.scatterplot(data=self.X, x=x,
                        y=y, hue=self.y, ax=ax[0, 0])
        sns.scatterplot(data=self.X, x=x,
                        y=y, hue=self.predict(self.X), ax=ax[0, 1])
        sns.scatterplot(data=self.test['X'], x=x,
                        y=y, hue=self.test['y'], ax=ax[1, 0])
        sns.scatterplot(data=self.test['X'], x=x,
                        y=y, hue=self.predict(self.test['X']), ax=ax[1, 1])
        ax[0, 0].set_title('Actual Value on Entire Data', fontsize=20)
        ax[0, 1].set_title('Prediction Value on Entire Data', fontsize=20)
        ax[0, 1].legend(title=self.test['y'].name)
        ax[1, 0].set_title('Actual Value on Test Data', fontsize=20)
        ax[1, 1].set_title('Prediction Value on Test Data', fontsize=20)
        ax[1, 1].legend(title=self.test['y'].name)
        plt.show()
