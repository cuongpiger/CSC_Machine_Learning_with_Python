import matplotlib.pyplot as plt
import seaborn as sns

class MySeaborn:
    def __init__(self, data):
        self.data = data
        
    def regplot(self, x, y):
        sns.regplot(data=self.data, x=x, y=y)
        plt.show()
        
    def residplot(self, x, y):
        sns.residplot(self.data[x], self.data[y])
        plt.show()
        
        
def visualPredictVsActual(y_predict, y_actual, x_scale, y_scale):
    plt.figure(figsize=(7, 7))
    plt.scatter(y_predict, y_actual)
    plt.xlabel('Model predictions')
    plt.ylabel('Actual value')
    plt.plot(x_scale, y_scale, 'k-', color='r')
    plt.show()