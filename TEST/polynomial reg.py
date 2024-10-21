import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

def liner():
    # Load the Iris dataset
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['petal length (cm)'] = iris.target  # Using petal length as the target variable

    # Using sepal length and sepal width as features
    X = df[['sepal length (cm)', 'sepal width (cm)']].values
    y = df['petal length (cm)'].values
    print(df.head())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_test_pred = model.predict(X_test)
    mse_test = mean_squared_error(y_test, y_test_pred)
    print(f'LINEAR REGRESSION\nMean Squared Error: {mse_test:.2f}')

    prediction_values = [float(input("Enter sepal length value: ")), 
                         float(input("Enter sepal width value: "))]
    prediction_values_reshaped = np.array(prediction_values).reshape(1, -1)
    predicted_value = model.predict(prediction_values_reshaped)
    print(f"The predicted petal length for sepal length {prediction_values[0]} cm and sepal width {prediction_values[1]} cm: {predicted_value[0]:.2f}")

    plt.figure(figsize=(6, 6))
    plt.scatter(X_test[:, 0], y_test, color='green', label='Test data (Sepal Length vs Petal Length)')
    plt.scatter(X_test[:, 0], y_test_pred, color='red', label='Predicted values', alpha=0.5)
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Petal Length (cm)')
    plt.title('Linear Regression on Iris Dataset')
    plt.legend()
    plt.grid(True)
    plt.show()

def poly():
    print("POLYNOMIAL REGRESSION")
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['petal length (cm)'] = iris.target 
    X = df[['sepal length (cm)', 'sepal width (cm)']].values
    y = df['petal length (cm)'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    poly = PolynomialFeatures(degree=2)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    y_pred = model.predict(X_test_poly)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R^2 Score: {r2:.2f}")

    prediction_values = [float(input("Enter sepal length value: ")), 
                         float(input("Enter sepal width value: "))]
    prediction_values_reshaped = np.array(prediction_values).reshape(1, -1)
    prediction_values_poly = poly.transform(prediction_values_reshaped)
    predicted_value = model.predict(prediction_values_poly)
    print(f"The predicted petal length for sepal length {prediction_values[0]} cm and sepal width {prediction_values[1]} cm: {predicted_value[0]:.2f}")

    plt.figure(figsize=(10, 6))
    plt.scatter(X_test[:, 0], y_test, color='green', label='Test data (Sepal Length vs Petal Length)')
    
    X_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 300).reshape(-1, 1)
    X_range_poly = poly.transform(np.hstack((X_range, np.zeros_like(X_range)))) 
    y_range_pred = model.predict(X_range_poly)
    plt.plot(X_range, y_range_pred, color='red', label='Polynomial fit')

    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Petal Length (cm)')
    plt.title('Polynomial Regression on Iris Dataset')
    plt.legend()
    plt.show()


liner()
poly()
