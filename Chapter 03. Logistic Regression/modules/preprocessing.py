import pandas_profiling as pp

class CPreprocessing:
    def __init__(self, data):
        self.data = data
        
    def eda(self):
        return pp.ProfileReport(self.data)