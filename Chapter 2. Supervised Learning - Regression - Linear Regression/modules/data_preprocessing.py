import pandas as pd
import numpy as np
import pandas_profiling as pp
from sklearn.preprocessing import RobustScaler

def logNormalization(data, features):
    for feature in features:
        data[feature + ' log'] = np.log(data[feature])

    return data


def removeOutliers(data):
    Q1 = data.quantile(.25)
    Q3 = data.quantile(.75)
    IQR = Q3 - Q1

    return data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]


def eda(data):
    return pp.ProfileReport(data)

def scaler(data, option='standard'):
    columns = data.columns
    
    if option == 'robust':
        return pd.DataFrame(RobustScaler().fit_transform(data), columns=columns)
        