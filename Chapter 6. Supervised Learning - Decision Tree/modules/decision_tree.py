from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from IPython.display import Image
import pydotplus


class CDecisionTree:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def initModel(self):
        self.model = DecisionTreeClassifier()
        self.model.fit(self.X, self.y)

    def predict(self, X=None):
        if X is None:
            X = self.test['X']

        return self.model.predict(X)

    def visualTree(self):
        dot_data = tree.export_graphviz(
            self.model, out_file=None, feature_names=self.X.columns)
        graph = pydotplus.graph_from_dot_data(dot_data)

        return Image(graph.create_png())

    def exportFile(self, save_path):
        dot_data = tree.export_graphviz(
            self.model, out_file=None, feature_names=self.X.columns)
        graph = pydotplus.graph_from_dot_data(dot_data)

        if '.pdf' in save_path:
            graph.write_pdf(save_path)
        elif '.png' in save_path:
            graph.write_png(save_path)

        print('Done.')

    def exportDotData(self, path):
        try:
            with open(path, 'w') as file:
                file = tree.export_graphviz(
                    self.model, out_file=file, feature_names=self.X.columns)
                
                print('Success!')
        except:
            print('Error!')
