import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt


data = pd.read_csv('/home/cs-ai-03/adhwaith/TEST/play_tennis.csv')



X = data.drop('play', axis=1)
y = data['play']


# X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


clf = DecisionTreeClassifier(criterion='entropy')  # Use entropy to simulate ID3
clf.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = clf.predict(X_test)

# Step 6: Evaluate the model
print(f'Accuracy: {metrics.accuracy_score(y_test, y_pred)}')
print(f'Confusion Matrix:\n{metrics.confusion_matrix(y_test, y_pred)}')

# Optional: Visualize the decision tree
plt.figure(figsize=(12,8))
plot_tree(clf, feature_names=X.columns, class_names=y.unique(), filled=True)
plt.show()
from sklearn import metrics
print(f'Accuracy: {metrics.accuracy_score(y_test, y_pred)}')
print(f'Confusion Matrix:\n{metrics.confusion_matrix(y_test, y_pred)}')