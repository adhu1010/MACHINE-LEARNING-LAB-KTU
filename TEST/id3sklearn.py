import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

data = pd.read_csv('/home/cs-ai-03/adhwaith/TEST/play_tennis.csv')

print(data.head())

X = data[['outlook','temp','humidity','wind']]
y = data['play']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

plt.figure(figsize=(20,10))
plot_tree(clf, feature_names=X.columns, class_names=['yes', 'no'], filled=True)
plt.show()

