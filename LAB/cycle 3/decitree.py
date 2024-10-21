import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("/home/cs-ai-03/adhwaith/LAB/cycle3/Buy_Computer.csv")
data = data.drop(columns=['id'])

label_encoders = {}
for column in data.columns:
    if data[column].dtype == 'object':
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

X = data.drop(columns=['Buy_Computer'])
y = data['Buy_Computer']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X_train, y_train)

plt.figure(figsize=(20, 10))
plot_tree(clf,
          feature_names=X.columns.tolist(),
          class_names=['No', 'Yes'],
          filled=True,
          rounded=True,
          fontsize=10)
plt.title('Decision Tree Visualization using Entropy')
plt.show()

def get_user_input(features, label_encoders):
    user_input = {}
    for feature in features:
        value = input(f"Enter value for {feature}: ")
        if feature in label_encoders:
            user_input[feature] = label_encoders[feature].transform([value])[0]
        else:
            user_input[feature] = value
    return pd.DataFrame([user_input], columns=features)

features = X.columns.tolist()
user_instance = get_user_input(features, label_encoders)
predicted_class = clf.predict(user_instance)
class_names = ['No', 'Yes']
predicted_class_label = class_names[predicted_class[0]]
print(f"Predicted Class: {predicted_class_label}")