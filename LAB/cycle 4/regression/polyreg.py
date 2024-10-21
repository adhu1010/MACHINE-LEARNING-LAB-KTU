import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.datasets import fetch_california_housing

def liner():
    data = pd.read_csv("/home/cs-ai-03/adhwaith/ML  LAB/LAB/cycle 4/regression/IceCreamData.csv")
    X = data[['Temperature']].values  
    y = data['Revenue'].values  
    print(data.head())


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

 
    y_test_pred = model.predict(X_test)
    mse_test = mean_squared_error(y_test, y_test_pred)
    print(f'LINEAR REGRESSION\nMean Squared Error: {mse_test:.2f}')

    prediction_value = float(input("Enter the temperature value: "))
    prediction_value_reshaped = np.array(prediction_value).reshape(1, -1)
    predicted_value = model.predict(prediction_value_reshaped)
    print(f"The revenue value for {prediction_value}°C: {predicted_value[0]:.2f}")


    plt.figure(figsize=(6, 6))
    plt.scatter(X_test, y_test, color='green', label='Test data')
    plt.plot(X_test, y_test_pred, color='red', linewidth=2, label='Regression line')
    plt.scatter([prediction_value], predicted_value, color='pink', s=100, edgecolor='black', zorder=5, label='Prediction point')
    plt.xlabel('Temperature')
    plt.ylabel('Revenue')
    plt.title('Linear Regression')
    plt.legend()
    plt.grid(True)
    plt.show()

def multi():
    print("MULTIVARIATE REGRESSION")
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


    print("Enter the values for the following features:")
    user_input = []
    for feature in iris.feature_names:
        value = float(input(f"{feature}: "))
        user_input.append(value)
    
    user_input_scaled = scaler.transform(np.array(user_input).reshape(1, -1))
    user_prediction = model.predict(user_input_scaled)[0]
    print(f"Predicted species for the entered values: {user_prediction}")


    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_train['sepal length (cm)'], X_train['sepal width (cm)'], 
                          c=y_train.astype('category').cat.codes, cmap='viridis', label='Training Data')
    plt.colorbar(scatter, ticks=[0, 1, 2], label='Species (0: Setosa, 1: Versicolor, 2: Virginica)')
    plt.scatter(user_input[0], user_input[1], color='red', s=100, label='User Input', edgecolor='black')    
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Sepal Width (cm)')
    plt.title('Multivariate Regression on Iris Dataset')
    plt.legend()
    plt.show()




def poly():
    print("POLYNOMIAL REGRESSION")
    data = fetch_california_housing(as_frame=True)
    print(data.frame.head())
    X = data.frame['MedInc'].values 
    print(X)
    Xset=X.reshape(-1, 1)
    y = data.frame['MedHouseVal'].values  
    yset=y.reshape(-1,1)
    print(y)
    X_train, X_test, y_train, y_test = train_test_split(Xset, yset, test_size=0.2, random_state=42)

   
    poly = PolynomialFeatures(degree=3)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    y_pred = model.predict(X_test_poly)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R^2 Score: {r2:.2f}")


    prediction_value = float(input("Enter the average income value: "))
    prediction_value_reshaped = np.array(prediction_value).reshape(1, -1)
    prediction_value_poly = poly.transform(prediction_value_reshaped)
    predicted_value = model.predict(prediction_value_poly)
    print(f"The predicted_value for medInc{prediction_value}is{predicted_value}")



    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color='green', label='Test data')
    

    X_range = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
    X_range_poly = poly.transform(X_range)
    y_range_pred = model.predict(X_range_poly)


    plt.plot(X_range, y_range_pred, color='red', label='Polynomial fit')

    plt.xlabel('MedINC')
    plt.ylabel('MEDHOUSEVAL')
    plt.title('Polynomial Regression: cal_housing')
    plt.legend()
    plt.grid(True)
    plt.show()



liner()
multi()
poly()
