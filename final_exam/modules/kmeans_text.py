import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from modules.drawer import CDrawer

from typing import List

drawer = CDrawer()
vectorizer = TfidfVectorizer(stop_words='english')

class CKMeansText:
    def __init__(self, documents: List[str]):
        self.X = vectorizer.fit_transform(documents)
        self.model = None
        
    def recommendKValue(self, k_upper: int = 10):
        sum_of_squared_distances = []
        
        for k in range(1, k_upper + 1):
            model = KMeans(n_clusters=k, max_iter=200, n_init=10)
            model = model.fit(self.X)
            sum_of_squared_distances.append(model.inertia_)
            
        drawer.line(pd.Series(np.arange(1, k_upper + 1), name='K-values'), 
                    pd.Series(sum_of_squared_distances, name="WSSE values"), "The Elbow Method\nshowing the optimal k-values ❤️\n")

    def initModel(self, k_value: int = 2):
        """ Model building method for K-Means
        
        Args:
            k_value (int, optional): k-value for clustering. Defaults to 2.
        """
        self.model = KMeans(n_clusters=k_value, init='k-means++', max_iter=200, n_init=10)
        self.model.fit(self.X)

    def showClusterCenter(self):
        """ List the list of k cluster center points
        
        Returns:
            ndarray[(float, float)]: list of cluster center points  
        """
        if self.model is not None:
            return np.array([p for p in self.model.cluster_centers_])
        
    def showSampleLabels(self):
        """ Indicates which cluster the samples belong
        
        Returns:
            pd.DataFrame: a pd.DataFrame includes the original self.X dataset and a `label`
                column indicating which cluster the sample belongs.
        """
        return self.model.labels_