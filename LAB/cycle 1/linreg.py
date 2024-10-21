import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.linear_model import LinearRegression

def main():
    print("Enter your data with one feature and one target variable in CSV format.like:(Feature,Target)\n finish with an empty line:")
  
    n = int(input("Enter the number of data points: "))

    X_values = []
    y_values = []

    print("Enter the values:")
    for i in range(n):
        x = float(input(f"Enter value for X[{i}]: "))
        y = float(input(f"Enter value for y[{i}]: "))
        X_values.append(x)
        y_values.append(y)

        X = np.array(X_values).reshape(-1, 1)
        y = np.array(y_values).reshape(-1, 1)

    df = pd.DataFrame({'Col_0': X.flatten(), 'Col_1': y.flatten()})

    print("\nGenerated DataFrame:")
    print(df)


    model = LinearRegression()
    model.fit(X, y)
    try:
        predict_value = float(input("\nEnter a value to predict (for the feature): "))
        prediction = model.predict(np.array([[predict_value]]))
        print(f"Prediction for feature value {predict_value}: {prediction[0]}")
    except ValueError:
        print("Invalid input for prediction. Please enter a numeric value.")
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', label='Data points')

    X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_pred = model.predict(X_range)
    plt.plot(X_range, y_pred, color='red', linewidth=2, label='Regression line')
   

    plt.scatter([predict_value], prediction, color='green', s=100, edgecolor='black', zorder=5, label='Prediction point')

    plt.xlabel('Feature')
    plt.ylabel('Target')
    plt.title('Linear Regression')
    plt.legend()
    plt.show()


main()