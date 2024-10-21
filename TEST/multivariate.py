import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def iris_classification():
    print("Multivariate regression")
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = iris.target_names[iris.target]
    print(df.head())
    
    X = df.drop('species', axis=1)
    y = df['species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

 
    print("Enter the values for the following features:")
    user_input = []
    for feature in iris.feature_names:
        value = float(input(f"{feature}: "))
        user_input.append(value)

    user_input_scaled = scaler.transform(np.array(user_input).reshape(1, -1))


    user_prediction = model.predict(user_input_scaled)[0]
    print(f"Predicted Species for the entered values: {user_prediction}")
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_train['sepal length (cm)'], X_train['sepal width (cm)'], c=y_train.astype('category').cat.codes, cmap='viridis', label='Training Data')
    plt.colorbar(scatter, ticks=[0, 1, 2], label='Species (0: Setosa, 1: Versicolor, 2: Virginica)')
    plt.scatter(user_input[0], user_input[1], color='red', s=100, label='User Input', edgecolor='black')    
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Sepal Width (cm)')
    plt.title('Iris Dataset Classification with User Input')
    plt.legend()
    plt.show()
iris_classification()
