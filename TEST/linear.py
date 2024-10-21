import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = pd.read_csv("/home/cs-ai-03/juhaina/dataset/IceCreamData.csv")
X = data[['Temperature']].values  
y = data['Revenue'].values  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_test_pred = model.predict(X_test)
mse_test = mean_squared_error(y_test, y_test_pred)
print(f'Mean Squared Error : {mse_test:.2f}')
plt.figure(figsize=(6, 6))
plt.subplot(1, 1, 1)
plt.scatter(X_test, y_test, color='green', label='Test data')
plt.plot(X_test, y_test_pred, color='red', linewidth=2, label='Regression line')
plt.xlabel('Temperature')
plt.ylabel('Revenue')
plt.title('linear regression')
plt.legend()
plt.grid(True)
plt.show()