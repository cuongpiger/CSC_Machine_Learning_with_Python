import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch

from typing import Optional
class CDrawer:
    def scatter(self, a: pd.Series, b: pd.Series, title=""):
        plt.scatter(a, b)
        plt.xlabel(a.name, color='b', weight='bold')
        plt.ylabel(b.name, color='b', weight='bold')
        plt.title(title, fontsize=18, color='r', weight='bold')
        
        plt.show()
        
    def clusterScatter3d(self, a: pd.Series, b: pd.Series, c: pd.Series, groups: pd.Series, clusters: Optional[np.ndarray] = None, y_hat: Optional[np.ndarray] = None, title=""):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(a, b, c, cmap='rainbow', )
        
    def clusterScatter(self, a: pd.Series, b: pd.Series, groups: pd.Series, clusters: np.ndarray = None, y_hat: Optional[np.ndarray] = None, title=""):
        if clusters is not None:
            plt.scatter(clusters[:, 0], clusters[:, 1], marker="*", c='red', s=150, linewidths=5, zorder=10)
        
        plt.scatter(a, b, c=groups)
        
        if y_hat is not None:
            plt.scatter(y_hat[:, 0], y_hat[:, 1], marker='s', c='b')
        
        plt.xlabel(a.name, color='b', weight='bold')
        plt.ylabel(b.name, color='b', weight='bold')
        plt.title(title, fontsize=18, color='r', weight='bold')
        
        plt.show()
        
    def line(self, a: pd.Series, b: pd.Series, title=""):
        plt.plot(a, b, 'co-')
        plt.xlabel(a.name, color='b', weight='bold')
        plt.ylabel(b.name, color='b', weight='bold')
        plt.title(title, fontsize=18, color='r', weight='bold')
        
        plt.show()
        
    def dendrogram(self, df: pd.DataFrame, title=""):
        plt.title(title, fontsize=18, color='r', weight='bold')
        dend = sch.dendrogram(sch.linkage(df, method='ward'))
        
        plt.show()
        
