import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

from typing import List


class CApriori:
    def __init__(self, transactions: List[List[str]]):
        """ Initial construction

        Args:
            transactions (List[List[str]]): [description]
        """
        self.transactions: pd.DataFrame = self.__prepareData(transactions)
        self.model = None

    def __prepareData(self, transactions: List[List[str]]):
        transformer = TransactionEncoder()
        transformer.fit(transactions)
        transactions_transform = transformer.transform(transactions)
        
        df = pd.DataFrame(transactions_transform, columns=transformer.columns_)
        df = df.drop(['nan'], axis=1)

        return df

    def initModel(self, min_support: int = .3):
        if self.model is None:
            self.model = apriori(self.transactions, min_support=min_support, use_colnames=True)
            
        return pd.DataFrame(self.model).set_index(['itemsets'])
        
    def associationInfo(self, metric='confidence', min_threshold=.3):
        if self.model is not None:
            return association_rules(self.model, metric=metric, min_threshold=min_threshold)

