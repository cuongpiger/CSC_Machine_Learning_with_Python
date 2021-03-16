from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
import pandas as pd

def selectKBest(X, y):
    best_features = SelectKBest(score_func=f_regression, k='all')
    fit = best_features.fit(X, y)
    scores = pd.DataFrame(fit.scores_)
    columns = pd.DataFrame(X.columns)
    featureScores = pd.concat([columns, scores], axis=1)
    featureScores.columns = ['Feature', 'Score']
    featureScores.sort_values('Score', ascending=False, inplace=True)
    
    return featureScores