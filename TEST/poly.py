import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

iris = load_iris()
X = iris.data[:, :1]  
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

degree = 2  
poly = PolynomialFeatures(degree=degree)
X_poly = poly.fit_transform(X_train)


model = LinearRegression()
model.fit(X_poly, y_train)

X_test_poly = poly.transform(X_test)
y_pred = model.predict(X_test_poly)


plt.scatter(X, y, color='blue', label='Data points')
x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
x_range_poly = poly.transform(x_range)
y_range = model.predict(x_range_poly)
plt.plot(x_range, y_range, color='red', label='Polynomial fit')
plt.xlabel('Sepal Length')
plt.ylabel('Species')
plt.title('Polynomial Regression on Iris Dataset')
plt.legend()
plt.show()
