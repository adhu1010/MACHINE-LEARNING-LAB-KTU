import math

import numpy as np
import pandas as pd


class DecisionTreeID3:
    def __init__(self):
        self.tree = None

    def entropy(self, data):
        target = data.iloc[:, -1]
        values, counts = np.unique(target, return_counts=True)
        entropy = 0
        for i in range(len(values)):
            probability = counts[i] / np.sum(counts)
            entropy += -probability * np.log2(probability)
        return entropy

    def information_gain(self, data, feature):
        total_entropy = self.entropy(data)
        values, counts = np.unique(data[feature], return_counts=True)
        weighted_entropy = 0
        for i in range(len(values)):
            subset = data[data[feature] == values[i]]
            weighted_entropy += (counts[i] / np.sum(counts)) * self.entropy(subset)
        return total_entropy - weighted_entropy

    def best_feature(self, data):
        features = data.columns[:-1]
        information_gains = [self.information_gain(data, feature) for feature in features]
        best_feature_index = np.argmax(information_gains)
        return features[best_feature_index]

    def fit(self, data):
        target = data.iloc[:, -1]
        if len(np.unique(target)) == 1:
            return np.unique(target)[0]

        if len(data.columns) == 1:
            return target.mode()[0]

        best_feature = self.best_feature(data)
        tree = {best_feature: {}}

        for value in np.unique(data[best_feature]):
            subset = data[data[best_feature] == value].drop(columns=[best_feature])
            subtree = self.fit(subset)
            tree[best_feature][value] = subtree

        self.tree = tree
        return tree

    def predict(self, query, tree=None):
        if tree is None:
            tree = self.tree

        for attribute in query:
            if attribute in tree:
                try:
                    subtree = tree[attribute][query[attribute]]
                    if isinstance(subtree, dict):
                        return self.predict(query, subtree)
                    else:
                        return subtree
                except KeyError:
                    return None

    def print_tree(self, tree=None, indent=""):
        if tree is None:
            tree = self.tree
import math

import numpy as np
import pandas as pd


class DecisionTreeID3:
    def __init__(self):
        self.tree = None

    def entropy(self, data):
        target = data.iloc[:, -1]
        values, counts = np.unique(target, return_counts=True)
        entropy = 0
        for i in range(len(values)):
            probability = counts[i] / np.sum(counts)
            entropy += -probability * np.log2(probability)
        return entropy

    def information_gain(self, data, feature):
        total_entropy = self.entropy(data)
        values, counts = np.unique(data[feature], return_counts=True)
        weighted_entropy = 0
        for i in range(len(values)):
            subset = data[data[feature] == values[i]]
            weighted_entropy += (counts[i] / np.sum(counts)) * self.entropy(subset)
        return total_entropy - weighted_entropy

    def best_feature(self, data):
        features = data.columns[:-1]
        information_gains = [self.information_gain(data, feature) for feature in features]
        best_feature_index = np.argmax(information_gains)
        return features[best_feature_index]

    def fit(self, data):
        target = data.iloc[:, -1]
        if len(np.unique(target)) == 1:
            return np.unique(target)[0]

        if len(data.columns) == 1:
            return target.mode()[0]

        best_feature = self.best_feature(data)
        tree = {best_feature: {}}

        for value in np.unique(data[best_feature]):
            subset = data[data[best_feature] == value].drop(columns=[best_feature])
            subtree = self.fit(subset)
            tree[best_feature][value] = subtree

        self.tree = tree
        return tree

    def predict(self, query, tree=None):
        if tree is None:
            tree = self.tree

        for attribute in query:
            if attribute in tree:
                try:
                    subtree = tree[attribute][query[attribute]]
                    if isinstance(subtree, dict):
                        return self.predict(query, subtree)
                    else:
                        return subtree
                except KeyError:
                    return None

    def print_tree(self, tree=None, indent=""):
        if tree is None:
            tree = self.tree

        if isinstance(tree, dict):
            for key, value in tree.items():
                print(indent + str(key) + "?")
                for sub_key, sub_value in value.items():
                    print(indent + "  " + str(key) + " = " + str(sub_key) + " ->")
                    self.print_tree(sub_value, indent + "    ")
        else:
            print(indent + "Predict: " + str(tree))

file_path = '/home/cs-ai-03/adhwaith/TEST/diabetes.csv' 
df = pd.read_csv(file_path)

df = df.iloc[:, 1:].applymap(str)

id3_tree = DecisionTreeID3()
tree = id3_tree.fit(df)

print("Decision Tree:")
id3_tree.print_tree()

query = {}
for column in df.columns[:-1]:
    value = input(f"Enter value for {column}: ")
    query[column] = value

prediction = id3_tree.predict(query)
print("\nPrediction for query {}: {}".format(query, prediction))
