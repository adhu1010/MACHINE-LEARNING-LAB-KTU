import pandas as pd
import numpy as np

train_data_m = pd.read_csv("/home/cs-ai-03/adhwaith/LAB/cycle3/Buy_Computer.csv")
train_data_m = train_data_m.drop(columns=['id'])

def calc_total_entropy(train_data, label, class_list):
    total_row = train_data.shape[0]
    total_entr = 0
   
    for c in class_list:
        total_class_count = train_data[train_data[label] == c].shape[0]
        total_class_entr = - (total_class_count/total_row)*np.log2(total_class_count/total_row)
        total_entr += total_class_entr
   
    return total_entr

def calc_entropy(feature_value_data, label, class_list):
    class_count = feature_value_data.shape[0]
    entropy = 0
   
    for c in class_list:
        label_class_count = feature_value_data[feature_value_data[label] == c].shape[0]
        entropy_class = 0
        if label_class_count != 0:
            probability_class = label_class_count/class_count
            entropy_class = - probability_class * np.log2(probability_class)
        entropy += entropy_class
    return entropy

def calc_info_gain(feature_name, train_data, label, class_list):
    feature_value_list = train_data[feature_name].unique()
    total_row = train_data.shape[0]
    feature_info = 0.0
   
    for feature_value in feature_value_list:
        feature_value_data = train_data[train_data[feature_name] == feature_value]
        feature_value_count = feature_value_data.shape[0]
        feature_value_entropy = calc_entropy(feature_value_data, label, class_list)
        feature_value_probability = feature_value_count/total_row
        feature_info += feature_value_probability * feature_value_entropy
       
    return calc_total_entropy(train_data, label, class_list) - feature_info

def find_most_informative_feature(train_data, label, class_list):
    feature_list = train_data.columns.drop(label)
    max_info_gain = -1
    max_info_feature = None
   
    for feature in feature_list:
        feature_info_gain = calc_info_gain(feature, train_data, label, class_list)
        if max_info_gain < feature_info_gain:
            max_info_gain = feature_info_gain
            max_info_feature = feature
           
    return max_info_feature

def generate_sub_tree(feature_name, train_data, label, class_list):
    feature_value_count_dict = train_data[feature_name].value_counts(sort=False)
    tree = {}    
    for feature_value, count in feature_value_count_dict.items():
        feature_value_data = train_data[train_data[feature_name] == feature_value]
        assigned_to_node = False
        for c in class_list:
            class_count = feature_value_data[feature_value_data[label] == c].shape[0]
           
            if class_count == count:
                tree[feature_value] = c
                train_data = train_data[train_data[feature_name] != feature_value]
                assigned_to_node = True
        if not assigned_to_node:
            tree[feature_value] = "?"
           
    return tree, train_data

def make_tree(root, prev_feature_value, train_data, label, class_list):
    if train_data.shape[0] != 0:
        max_info_feature = find_most_informative_feature(train_data, label, class_list)
        tree, train_data = generate_sub_tree(max_info_feature, train_data, label, class_list)
        next_root = None
       
        if prev_feature_value is not None:
            root[prev_feature_value] = dict()
            root[prev_feature_value][max_info_feature] = tree
            next_root = root[prev_feature_value][max_info_feature]
        else:
            root[max_info_feature] = tree
            next_root = root[max_info_feature]
       
        for node, branch in list(next_root.items()):
            if branch == "?":
                feature_value_data = train_data[train_data[max_info_feature] == node]
                make_tree(next_root, node, feature_value_data, label, class_list)

def id3(train_data_m, label):
    train_data = train_data_m.copy()
    tree = {}
    class_list = train_data[label].unique()
    make_tree(tree, None, train_data, label, class_list)
    return tree

tree = id3(train_data_m, 'Buy_Computer')


def print_tree(tree, indent=""):
    if isinstance(tree, dict):
        for key, value in tree.items():
            print(f"{indent}|---{key}:")
            print_tree(value, indent + "  ")
    else:
        print(f"{indent}{tree}")

def predict(tree, instance):
    if not isinstance(tree, dict):
        return tree
    else:
        root_node = next(iter(tree))
        feature_value = instance[root_node]
        if feature_value in tree[root_node]:
            return predict(tree[root_node][feature_value], instance)
        else:
            return None

def evaluate(tree, test_data_m, label):
    correct_predict = 0
    wrong_predict = 0
    for index, row in test_data_m.iterrows():
        result = predict(tree, test_data_m.iloc[index])
        if result == test_data_m[label].iloc[index]:
            correct_predict += 1
        else:
            wrong_predict += 1
    accuracy = correct_predict / (correct_predict + wrong_predict)
    return accuracy

def predict_class(tree, instance):
    return predict(tree, instance)

def get_user_input(features):
    user_input = {}
    for feature in features:
        user_input[feature] = input(f"Enter value for {feature}: ")
    return pd.Series(user_input)

test_data_m = pd.read_csv("/home/cs-ai-03/juhaina/Buy_Computer.csv")
test_data_m = test_data_m.drop(columns=['id'])

accuracy = evaluate(tree, test_data_m, 'Buy_Computer')
print(f"Accuracy: {accuracy}")

print("\nDecision Tree:")
print_tree(tree)

features = ['age', 'income', 'student', 'credit_rating']  
user_instance = get_user_input(features)

predicted_class = predict_class(tree, user_instance)
print(f"\nTest Instance:\n{user_instance}")
print(f"Predicted Class: {predicted_class}")

def find_closest_instance(user_instance, dataset, features):
    for index, row in dataset.iterrows():
        if all(row[feature] == user_instance[feature] for feature in features):
            return row
    return None

closest_instance = find_closest_instance(user_instance, test_data_m, features)
if closest_instance is not None:
    actual_class = closest_instance['Buy_Computer']
    print(f"Actual Class: {actual_class}")