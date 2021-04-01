import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
from modules.drawer import CDrawer

from typing import List, Optional

drawer = CDrawer()

class CKMeans:
    def __init__(self, X: pd.DataFrame):
        """ Constructor method

        Args:
            X (pd.DataFrame): training dataset.
        """
        self.X: pd.DataFrame = X
        self.model = None
        
    def initModel(self, k_value: int = 2):
        """ Model building method for K-Means
        
        Args:
            k_value (int, optional): k-value for clustering. Defaults to 2.
        """
        self.model = KMeans(n_clusters=k_value)
        self.model.fit(self.X)
        
    def predict(self, data: pd.DataFrame or np.ndarray) -> Optional[np.ndarray]:
        if self.model is not None:
            return self.model.predict(data)
        else:
            print('The model has not been trained!')
            return None
        
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
                        pd.Series(lst_wsse, name="WSSE values"), "The Elbow Method\nshowing the optimal k-values ❤️\n")
            
        return pd.DataFrame([(k + 1, v) for k, v in enumerate(lst_wsse)], columns=['k-value', 'wsse-value']).set_index(['k-value'])
        
    def showClusterCenter(self):
        """ List the list of k cluster center points
        
        Returns:
            ndarray[(float, float)]: list of cluster center points  
        """
        if self.model is not None:
            return np.array([(p[0], p[1]) for p in self.model.cluster_centers_])
        
    def showSampleLabels(self):
        """ Indicates which cluster the samples belong
        
        Returns:
            pd.DataFrame: a pd.DataFrame includes the original self.X dataset and a `label`
                column indicating which cluster the sample belongs.
        """
        res: pd.DataFrame = self.X.copy()
        res['label'] = self.model.labels_
        
        return res
    
    def visualModelTraining(self):
        if self.model is not None:
            if self.X.shape[1] == 2:
                cluster_points = self.showClusterCenter()
                groups = pd.Series(self.model.labels_)
                drawer.clusterScatter(self.X.iloc[:, 0], self.X.iloc[:, 1], groups, cluster_points, title="K-Means with 2 features dataset ❤️\n")