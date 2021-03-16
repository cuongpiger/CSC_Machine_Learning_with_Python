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