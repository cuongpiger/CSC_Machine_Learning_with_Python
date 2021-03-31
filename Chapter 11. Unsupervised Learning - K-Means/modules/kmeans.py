import pandas as pd


class CKMeans:
    def __init__(self, X: pd.DataFrame):
        """ Constructor method

        Args:
            X (pd.DataFrame): [training dataset]
        """
        self.X: pd.DataFrame = X
        
    def initModel(self, k_value: int = 2):
        """ Model building method for K-Means
        
        Args:
            k_value (int, optional): [k-value for clustering]. Defaults to 2.
        """
        pass
    
    def selectKValue(self):
        """ Method used to visualize and suggest the best k-value
        """
        pass