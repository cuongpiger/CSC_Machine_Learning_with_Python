import numpy as np
import pandas as pd
import pandas_profiling as pp
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, chi2, f_regression
from sklearn.model_selection import train_test_split

from typing import List

class CPreprocessing:
    def unique(self, data):
        """ Dùng để kiểm tra số giá trị độc nhất trên các feature của `data`
        Args:
            data (DataFrame): toàn bộ dữ liệu đầu vào 
        """
        res = pd.DataFrame(data.nunique(), columns=['Number of unique values'])
        res['Values'] = [data[column].unique() for column in data.columns]
        
        return res
    
    def duplicate(self, data):
        """ Kiểm tra `data` có chứa các sample bị duplicate hay ko
        Args:
            data (DataFrame): toàn bộ dữ liệu đầu vào
        """
        
        dups = data.duplicated()
        
        if dups.any() == True:
            return data[dups]
        
        return 'The data does not contain any duplicate sample.'
    
    def countOutliers(self, data, features):
        tmp = []
        
        for feature in features:
            data_mean, data_std = np.mean(data[feature]), np.std(data[feature])
            # define outliers
            cut_off = data_std * 3
            lower, upper = data_mean - cut_off, data_mean + cut_off
            # identify outliers
            outliers = [x for x in data[feature] if x < lower or x > upper]
            
            tmp.append([feature, len(outliers), data.shape[0] - len(outliers)])
            
        return pd.DataFrame(tmp, columns=['Feature', 'Number of outliers', 'Number of non-outiers'])
    
    def eda(self, data):
        return pp.ProfileReport(data)
    
    def removeOutliers(self, data):
        df = data.copy()
        
        for feature in df.columns:
            data_mean, data_std = np.mean(df[feature]), np.std(df[feature])
            # define outliers
            cut_off = data_std * 3
            lower, upper = data_mean - cut_off, data_mean + cut_off
            # identify outliers
            df = df[(df[feature] >= lower) & (df[feature] <= upper)]
            
        return df
    
    def selectCategoricalFeature(self, X, y):
        X_train, _, y_train, _ = train_test_split(X, y)
        fs = SelectKBest(score_func=chi2, k='all')
        fs.fit(X_train, y_train)
        for i, feature in enumerate(X.columns):
            print('Feature %s: %f' % (feature, fs.scores_[i]))
        # plot the scores
        plt.bar(X.columns, fs.scores_)
        plt.show()
        
    def selectNumericalFeature(self, X, y):
        X_train, _, y_train, _ = train_test_split(X, y)
        fs = SelectKBest(score_func=f_regression, k='all')
        fs.fit(X_train, y_train)
        for i, feature in enumerate(X.columns):
            print('Feature %s: %f' % (feature, fs.scores_[i]))
        # plot the scores
        plt.bar(X.columns, fs.scores_)
        plt.show()