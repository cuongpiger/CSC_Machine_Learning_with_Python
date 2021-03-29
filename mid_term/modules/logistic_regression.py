from sklearn import model_selection
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class CLogisticRegression:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def prepareData(self, test_size=None, random_state=None):
        self.train, self.test = {}, {}
        self.train['X'], self.test['X'], self.train['y'], self.test['y'] = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)
            
    def initModel(self):
        self.model = LogisticRegression()
        self.model.fit(self.train['X'], self.train['y'])

    def predict(self, X=None):
        if X is None:
            X = self.test['X']
            
        return self.model.predict(X)
    
    def accuracy(self, X=None):
        return accuracy_score(self.test['y'], self.predict(X))
    
    def confusionMatrix(self, X=None):
        target_classes = self.test['y'].value_counts().index
        cm = confusion_matrix(self.test['y'], self.predict(X))
        cm_df = pd.DataFrame(cm, index=target_classes, columns=target_classes)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_df, annot=True, fmt='g', cmap='Blues')
        plt.title('Logistic Regression\nAccuracy: {0:.3f}\nNumber of samples: {1}'.format(self.accuracy(), self.test['y'].shape[0]))
        plt.ylabel('Actual values')
        plt.xlabel('Predicted values')
        plt.yticks(rotation=0) 
        plt.show()
        return (pd.DataFrame(self.test['y'].value_counts()))
    
    def precisionRecall(self, X=None):
        print(classification_report(self.test['y'], self.predict(X), target_names=self.test['y'].value_counts().index))
        
    def kfold(self, n_splits=10):
        model = LogisticRegression()
        res = model_selection.cross_val_score(model, self.X, self.y, cv=KFold(n_splits=n_splits))
        
        print('Accuracy - mean: {0:.3f}%, std: {1:.3f}%'.format(res.mean()*100., res.std()*100.))