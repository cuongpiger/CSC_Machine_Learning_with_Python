import pandas_profiling as pp
import pandas as pd

class CPreprocessing:
    def __init__(self, data):
        self.data = data
        
    def eda(self):
        return pp.ProfileReport(self.data)
    
    def encoding(self, option='dummy', drop_first=False):
        if option == 'dummy':
            return pd.get_dummies(self.data, drop_first=drop_first)