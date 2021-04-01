import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from modules.drawer import CDrawer

from typing import List, Optional

drawer = CDrawer()


class CHierarchical:
    def __init__(self, X: pd.DataFrame):
        self.X = X
        self.model = None

    def initModel(self, k_value: int = 2):
        self.model = AgglomerativeClustering(
            n_clusters=k_value, affinity='euclidean', linkage='ward')
        self.model.fit(self.X)

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
                groups = pd.Series(self.model.labels_)
                drawer.clusterScatter(self.X.iloc[:, 0], self.X.iloc[:, 1], groups, title="Hierarchical with 2 features dataset ❤️\n")
