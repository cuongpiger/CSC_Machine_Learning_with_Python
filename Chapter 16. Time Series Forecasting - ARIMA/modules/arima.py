from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima
import pandas as pd
import matplotlib.pyplot as plt

class CArima:
    def __init__(self, X: pd.DataFrame):
        self.X: pd.DataFrame = X
        self.model = None
        
    def selectBestParameters(self):
        if self.model is None:
            self.model = auto_arima(self.X, start_p=2, start_q=2, max_p=5, max_q=5, m=12, start_P=1, seasonal=True, d=1, D=1, trace=True, error_action="ignore", suppress_warnings=True, stepwise=True)
            
    def initModel(self, training_data: pd.DataFrame):
        if self.model is not None:
            self.model.fit(training_data)
            
    def predict(self, prediected_data: pd.DataFrame):
        return pd.DataFrame(self.model.predict(n_periods=len(prediected_data)), index=prediected_data.index, columns=['Prediction'])
