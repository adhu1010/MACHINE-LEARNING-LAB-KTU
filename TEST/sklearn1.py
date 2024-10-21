import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier    
from sklearn.metrics import accuracy_score

df=pd.read_csv("Play Tennis.csv")
value=['Outlook','Temprature','Humidity','Wind']
string_to_int= preprocessing.LabelEncoder()                  
df=df.apply(string_to_int.fit_transform) 
feature_cols = ['Outlook','Temprature','Humidity','Wind']
X = df[feature_cols ]                            
y = df.Play_Tennis
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)


                         # import the classifier
classifier =DecisionTreeClassifier(criterion="entropy", random_state=100)     # create a classifier object
classifier.fit(X_train, y_train) 
y_pred= classifier.predict(X_test) 

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
data_p=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})  