import pandas as pd
import numpy as np
import math


file_path = '/home/cs-ai-03/adhwaith/TEST/customer_churn_dataset-testing-master.csv' 
df = pd.read_csv(file_path)

df = df.iloc[:, 1:].applymap(str)

def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    entropy = sum([(-counts[i]/sum(counts)) * math.log2(counts[i]/sum(counts)) for i in range(len(elements))])
    return entropy

def info_gain(data, split_attribute_name, target_name="Churn"):
    total_entropy = entropy(data[target_name])
    values, counts = np.unique(data[split_attribute_name], return_counts=True)
    weighted_entropy = sum([(counts[i]/sum(counts)) * entropy(data.where(data[split_attribute_name]==values[i]).dropna()[target_name]) for i in range(len(values))])
    information_gain = total_entropy - weighted_entropy
    return information_gain


def id3(data, original_data, features, target_attribute_name="Churn", parent_node_class=None):

    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]
    

    elif len(data) == 0:
        return np.unique(original_data[target_attribute_name])[np.argmax(np.unique(original_data[target_attribute_name], return_counts=True)[1])]
    

    elif len(features) == 0:
        return parent_node_class
    

    else:
        parent_node_class = np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]
        

        item_values = [info_gain(data, feature, target_attribute_name) for feature in features]
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]
        

        tree = {best_feature: {}}
        

        features = [i for i in features if i != best_feature]
        

        for value in np.unique(data[best_feature]):
            sub_data = data.where(data[best_feature] == value).dropna()
            subtree = id3(sub_data, original_data, features, target_attribute_name, parent_node_class)
            tree[best_feature][value] = subtree
        
        return tree

features = ['Age', 'Gender', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Subscription Type', 'Contract Length', 'Total Spend', 'Last Interaction']
decision_tree = id3(df, df, features)
print(decision_tree)
