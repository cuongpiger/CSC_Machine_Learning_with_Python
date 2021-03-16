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
    
def visualDistributionPlot(model):
    plt.figure(figsize=(14,6))
    plt.subplot(1,2,1)
    ax1 = sns.distplot(model.train['y'], hist=False, color="r", 
                    label="Actual Train Values")
    sns.distplot(model.predict(model.train['X']), hist=False, color="b", 
                label="Predicted Train Values", ax=ax1)

    plt.subplot(1,2,2)
    ax2 = sns.distplot(model.test['y'], hist=False, color="r", 
                    label="Actual Test Values")
    sns.distplot(model.predict(model.test['X']), hist=False, color="b", 
                label="Predicted Test Values" , ax=ax2)

    plt.show()
    
def heatmap(corr_mat):
    plt.figure(figsize=(15, 7))
    graph = sns.heatmap(corr_mat, cmap='RdYlGn', annot=True) # cờ `annot` dùng để hiển thị số trong từng cell của heatmap
    plt.show()