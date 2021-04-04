from sklearn.decomposition import PCA
import pandas as pd

class CPCA:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def initModel(self, param=2):
        if type(param) == int:
            self.pca = PCA(n_components=param)
            data = self.pca.fit_transform(self.data)

            return pd.DataFrame(data=data, columns=['pa_' + str(i) for i in range(1, param + 1)])
        elif type(param) == float:
            self.pca = PCA(param)
            self.pca.fit(self.data)

    def transform(self, data: pd.DataFrame):
        return self.pca.transform(data)


        