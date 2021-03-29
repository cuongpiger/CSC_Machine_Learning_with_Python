import matplotlib.pyplot as plt
import seaborn as sns

class CDrawer:
    def boxplotGrid(self, shape, data):
        fig, axs = plt.subplots(shape[0], shape[1], figsize=(10, 6))
        
        for i, column in enumerate(data.columns):
            row, col = i // shape[1], i % shape[1]
            axs[row, col].boxplot(data[column], 0, 'rD')
            axs[row, col].set_title(column)
            
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.05, top=0.9, hspace=0.4, wspace=0.3)
            
        plt.show()