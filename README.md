# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the Iris dataset and separate features and labels.
2.plit the data into training and testing sets.
3.Train an SGD classifier on the training data.
4.Predict on the test set and evaluate using accuracy and a classification report
## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Syed Samieudeen
RegisterNumber: 212225240165  
*/
```
```
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = SGDClassifier(max_iter=1000, tol=1e-3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

## Output:
<img width="1121" height="255" alt="554517057-74aba0c5-b031-4f4a-970f-eb54a70b4ff8" src="https://github.com/user-attachments/assets/7f86de4e-524d-45c7-bd3d-1ce58670316f" />


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
