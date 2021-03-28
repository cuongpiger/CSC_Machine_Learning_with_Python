import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori

class CApriori:
    def __init__(self, data):
        self.data = data
        
    def prepareData(self):
        pass
    
    def initModel(self):
        self.model = TransactionEncoder()
        return pd.DataFrame(self.model.fit(self.data).transform(self.data), columns=self.model.columns_)
    
    def transform(self, data=None):
        if data is None:
            data = self.data
            
        return pd.DataFrame(self.model.transform(data), columns=self.model.columns_)