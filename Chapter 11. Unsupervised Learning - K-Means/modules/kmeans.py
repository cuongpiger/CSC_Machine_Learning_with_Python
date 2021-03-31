import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
from modules.drawer import CDrawer

from typing import List

drawer = CDrawer()

class CKMeans:
    def __init__(self, X: pd.DataFrame):
        """ Constructor method

        Args:
            X (pd.DataFrame): training dataset.
        """
        self.X: pd.DataFrame = X
        
    def initModel(self, k_value: int = 2):
        """ Model building method for K-Means
        
        Args:
            k_value (int, optional): k-value for clustering. Defaults to 2.
        """
        self.model = KMeans(n_clusters=k_value)
        self.model.fit(self.X)
    
    def recommendKValue(self, k_upper: int = 10, visual: bool = False):
        """ Method used to visualize and suggest the best k-value based on Elbow Method
        
        Args:
            k_upper (int, optional): the upper bound of k-value. Default to 10. 
            visual (bool, optional): a flag used to visualize or not. Defaults to False.
        
        Returns:
            List[(int, int)]: p1 is the k-value and p2 is the corresponding msse value
        """
        
        lst_wsse: List[int] = []
        
        for k in range(1, k_upper + 1):
            model = KMeans(n_clusters=k)
            model.fit(self.X)
            lst_wsse.append(sum(np.min(cdist(self.X, model.cluster_centers_, 'euclidean'), axis=1))/self.X.shape[0])
            
        if visual:
            drawer.line(pd.Series(np.arange(1, k_upper + 1), name='K-values'), 
                        pd.Series(lst_wsse, name="WSSE values"), "The Elbow Method\nshowing the optimal k-values")
            
        return pd.DataFrame([(k + 1, v) for k, v in enumerate(lst_wsse)], columns=['k-value', 'wsse-value']).set_index(['k-value'])
        