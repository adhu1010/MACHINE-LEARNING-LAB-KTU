import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.datasets import fetch_california_housing
def poly():
    print("POLYNOMIAL REGRESSION")
    data = pd.read_csv("/home/cs-ai-03/adhwaith/ML LAB/LAB/cycle 4/regression/IceCreamData.csv")
    X = data[['Temperature']].values  
    y = data['Revenue'].values  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create polynomial features
    poly = PolynomialFeatures(degree=2)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # Fit the model
    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    # Predictions
    y_pred = model.predict(X_test_poly)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R^2 Score: {r2:.2f}")

    # User input for prediction
    prediction_value = float(input("Enter the temperature value: "))
    prediction_value_reshaped = np.array([[prediction_value]])  # Reshape to 2D array for single feature
    prediction_value_poly = poly.transform(prediction_value_reshaped)
    predicted_value = model.predict(prediction_value_poly)
    
    # Access the first element correctly
    print(f"The revenue value for {prediction_value}°C: {predicted_value[0]:.2f}")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color='green', label='Test data')
    
    # Generate range for temperature values for plotting the polynomial curve
    X_range = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
    X_range_poly = poly.transform(X_range)
    y_range_pred = model.predict(X_range_poly)

    # Plot the polynomial regression line
    plt.plot(X_range, y_range_pred, color='red', label='Polynomial fit')

    plt.xlabel('Temperature')
    plt.ylabel('Revenue')
    plt.title('Polynomial Regression: Ice Cream Sales')
    plt.legend()
    plt.grid(True)
    plt.show()

# Call the function to execute
poly()
