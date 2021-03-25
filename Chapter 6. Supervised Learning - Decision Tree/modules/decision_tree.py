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
