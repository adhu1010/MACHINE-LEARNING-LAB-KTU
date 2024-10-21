import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


iris = load_iris()
X = iris.data 
y = iris.target 


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)
clusters = kmeans.labels_


plt.figure(figsize=(10, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis', marker='o', edgecolor='k', s=100)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.title('K-means Clustering of Iris Dataset')
plt.xlabel('Feature 1 (Standardized)')
plt.ylabel('Feature 2 (Standardized)')
plt.legend()
plt.show()


results = pd.DataFrame({'Actual': y, 'Predicted': clusters})
print(results.head(100))


def predict_cluster():
    print("Enter the following features for prediction:")
    features = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
    user_input = []
    
    for feature in features:
        value = float(input(f"{feature}: "))
        user_input.append(value)

    user_input_scaled = scaler.transform([user_input])

    predicted_cluster = kmeans.predict(user_input_scaled)
    print(f"The predicted cluster for the provided input is: {predicted_cluster[0]}")


predict_cluster()
