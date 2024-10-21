import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd

iris=load_iris()
data=iris.data
target=iris.target
target_names=iris.target_names
feature_names=iris.feature_names

df=pd.DataFrame(data,columns=feature_names)
df['species']=target
plt.figure(figsize=(12,8))

plt.subplot(2,2,1)
for species in range(len(target_names)):
    subset=df[df['species']==species]
    plt.scatter(subset[feature_names[0]],subset[feature_names[1]],label=target_names[species])
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.title('Sepal Length vs Sepal Width')
plt.legend()

plt.subplot(2,2,2)
for species in range(len(target_names)):
    subset=df[df['species']==species]
    plt.scatter(subset[feature_names[0]],subset[feature_names[2]],label=target_names[species])
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[2])
plt.title('Sepal Length vs Petal Length')
plt.legend()

plt.subplot(2,2,3)
for species in range(len(target_names)):
    subset=df[df['species']==species]
    plt.scatter(subset[feature_names[1]],subset[feature_names[2]],label=target_names[species])
plt.xlabel(feature_names[1])
plt.ylabel(feature_names[2])
plt.title('Sepal Width vs Petal Length')
plt.legend()

plt.subplot(2,2,4)
for species in range(len(target_names)):
    subset=df[df['species']==species]
    plt.scatter(subset[feature_names[2]],subset[feature_names[3]],label=target_names[species])
plt.xlabel(feature_names[2])
plt.ylabel(feature_names[3])
plt.title('Petal Length vs Petal Width')
plt.legend()

plt.tight_layout()
plt.show()