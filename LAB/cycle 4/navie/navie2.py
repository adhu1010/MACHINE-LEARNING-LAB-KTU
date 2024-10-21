import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from math import sqrt, pi, exp

url = "/home/cs-ai-03/adhwaith/LAB/cycle 4/navie/diabetes.csv"
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 
           'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
data = pd.read_csv(url, names=columns)
print(data.head())

X = data.drop('Outcome', axis=1).values
y = data['Outcome'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)

def summarize_by_class(X_train, y_train):
    separated = {0: [], 1: []}
    for i in range(len(y_train)):
        row = X_train[i]
        label = y_train[i]
        separated[label].append(row)
    
    summaries = {}
    for class_value, rows in separated.items():
        summaries[class_value] = [(np.mean(column), np.std(column)) for column in zip(*rows)]
    return summaries


def gaussian_probability(x, mean, std):
    exponent = exp(-((x - mean) ** 2 / (2 * std ** 2)))
    return (1 / (sqrt(2 * pi) * std)) * exponent


def calculate_class_probabilities(summaries, row):
    total_rows = sum([len(summaries[label]) for label in summaries])
    probabilities = {}
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = 1 
        for i in range(len(class_summaries)):
            mean, std = class_summaries[i]
            probabilities[class_value] *= gaussian_probability(row[i], mean, std)
    return probabilities

def predict(summaries, row):
    probabilities = calculate_class_probabilities(summaries, row)
    return max(probabilities, key=probabilities.get)


def get_predictions(summaries, X_test):
    predictions = []
    for row in X_test:
        predictions.append(predict(summaries, row))
    return predictions


def calculate_accuracy(y_test, y_pred):
    correct = 0
    for i in range(len(y_test)):
        if y_test[i] == y_pred[i]:
            correct += 1
    return correct / len(y_test)

def calculate_precision_recall(y_test, y_pred):
    true_positives = false_positives = false_negatives = 0
    
    for i in range(len(y_test)):
        if y_test[i] == 1 and y_pred[i] == 1:
            true_positives += 1
        elif y_test[i] == 0 and y_pred[i] == 1:
            false_positives += 1
        elif y_test[i] == 1 and y_pred[i] == 0:
            false_negatives += 1

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    return precision, recall


summaries = summarize_by_class(X_train, y_train)


y_pred = get_predictions(summaries, X_test)


accuracy = calculate_accuracy(y_test, y_pred)
precision, recall = calculate_precision_recall(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")

