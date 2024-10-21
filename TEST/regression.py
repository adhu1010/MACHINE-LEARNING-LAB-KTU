import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes





def single_variable_regression(X, y):
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    return model, y_pred


def multivariate_regression(X, y):
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    return model, y_pred

def polynomial_regression(X, y, degree=2):
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    y_pred = model.predict(X_poly)
    return model, y_pred, poly_features

def evaluate_model(y, y_pred):
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    return mse, r2

def plot_results(X, y, y_pred, title):
    plt.scatter(X, y, color='blue', label='Actual')
    plt.plot(X, y_pred, color='red', label='Predicted')
    plt.title(title)
    plt.xlabel('Features')
    plt.ylabel('Target')
    plt.legend()
    plt.show()


def main():
    
    diabetes =load_diabetes()
    data = pd.DataFrame(diabetes.data ,columns=diabetes.feature_names)
    print(data.head)
    data["target"]=diabetes.target
    
   

    y = data["target"]
    X = data[['bmi']]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    
   
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print('LinearRegression')
    print("Mean Squared Error:", mse)
    print("R-squared:", r2)


    X = data.drop(columns="target")
    y = data["target"] 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
 
    print("MultiVariate Regression")
    print("Mean Squared Error:", mse)
    print("R-squared:", r2)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)


    model = LinearRegression()
    model.fit(X_train, y_train)


    y_pred = model.predict(X_test)


    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("polynomial Regression")
    print("Mean Squared Error:", mse)
    print("R-squared:", r2)


if __name__ == "__main__":
    main()
