from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

import pandas as pd

class CHoltWinters:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def decomposition(self):
        res = seasonal_decompose(self.data, model='multiplicative')
        res.plot()
        plt.show()

    def initModel(self, training_data: pd.DataFrame):
        self.model = ExponentialSmoothing(training_data, seasonal='mul', seasonal_periods=12).fit()

    def predict(self, data: pd.DataFrame):
        return self.model.predict(start=data.index[0], end=data.index[-1])