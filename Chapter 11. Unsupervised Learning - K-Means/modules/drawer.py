import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class CDrawer:
    def scatter(self, a: pd.Series, b: pd.Series, title=""):
        plt.scatter(a, b)
        plt.xlabel(a.name)
        plt.ylabel(b.name)
        plt.title(title, fontsize=18, color='r')
        
        plt.show()
        
    def line(self, a: pd.Series, b: pd.Series, title=""):
        plt.plot(a, b, 'co-')
        plt.xlabel(a.name)
        plt.ylabel(b.name)
        plt.title(title, fontsize=18, color='r')
        
        plt.show()
