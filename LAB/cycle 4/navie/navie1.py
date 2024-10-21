import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, recall_score, precision_score


url = "/home/cs-ai-03/adhwaith/ML  LAB/LAB/cycle 4/navie/diabetes.csv"
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 
           'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
data = pd.read_csv(url, names=columns)
print(data.head())


x = data.drop('Outcome', axis=1) 
y = data['Outcome']  


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)


gnb = GaussianNB()
gnb.fit(x_train, y_train)


y_pred = gnb.predict(x_test)


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary')
recall = recall_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")


def user_input():
    print("Enter the following details:")
    input_data = []
    features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    
    for feature in features:
        value = float(input(f"{feature}: "))
        input_data.append(value)

    return np.array(input_data).reshape(1, -1)


user_data = user_input()
prediction = gnb.predict(user_data)
if prediction[0]==1:
    print("The person has diabetes ")
else:
    print("The person has not dagonosed with diabetes ")
