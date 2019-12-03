# TITANIC DATASET ANALYSIS
# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the datasets
train_dataset = pd.read_csv("train.csv")
test_dataset = pd.read_csv("test.csv")
test_pred = pd.read_csv("gender_submission.csv")

# Retrieving the "Survived" column from the train dataset and assign to 'y', the dependent variable
y = train_dataset["Survived"].values
# Dropping the "Survived" column from the train dataset for concatenation
train_dataset = train_dataset.drop(["Survived"], axis=1)
# Concatenation of the two datasets into the 'dataset' variable
dataset = pd.concat([train_dataset, test_dataset], ignore_index=True, axis=0)
# Dropping the useless columns from the new dataset
dataset = dataset.drop(["PassengerId", "Name", "Ticket", "Fare", "Cabin"], axis=1)

# Getting rid of the missing data
dataset["Embarked"].fillna(method='ffill', inplace=True)
dataset["Age"].fillna(dataset["Age"].median(), inplace=True)

# Encoding categorical data into dummy variables
# And avoiding the dummy variable trap (multicollinearity) Manually
sex_dummies = pd.get_dummies(dataset['Sex'], prefix='Sex')
sex_dummies = sex_dummies.iloc[:, 0]
embarked_dummies = pd.get_dummies(dataset['Embarked'], prefix='Embarked')
embarked_dummies = embarked_dummies.iloc[:, :2]
# Dropping original categorical columns from dataset and concatenating the new ones
dataset = dataset.drop(["Sex", "Embarked"], axis=1)
dataset = pd.concat([sex_dummies, embarked_dummies, dataset], axis=1)

# Independent variable assignation
X = dataset.iloc[:].values

# Splitting data into Training and Test sets manually and let y_test to the end
X_train = X[:891, :]
X_test = X[891:, :]
y_train = y[:]
y_test = test_pred.iloc[:, 1].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# *****KNN COMPROBRATION*****
# Fitting the classifier to the training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=9, metric='euclidean', p=2, n_jobs=-1)
classifier.fit(X_train, y_train)

# Predicting the test set results
y_pred = classifier.predict(X_test)

# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Accuracy
TP = cm[0][0]
TN = cm[1][1]
FP = cm[1][0]
FN = cm[0][1]
accuracy = ((TP+TN)/(TP+TN+FP+FN))
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = (2*precision*recall)/(precision+recall)
print("Parameters Comprobation")
print("The accuracy of the model created is {:0.4f}.".format(accuracy))
print("The precision of the model created is {:0.4f}.".format(precision))
print("The recall of the model created is {:0.4f}.".format(recall))
print("The F1 score of the model created is {:0.4f}.\n".format(f1_score))

# Principal Component Analysis (PCA)
from sklearn.decomposition import PCA
pca = PCA(n_components=4)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

# AFTER GETTING THE PCA, RERUN THIS FOR PLOTTING AFTER IT
# *****KNN COMPROBRATION*****
# Fitting the classifier to the training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=9, metric='euclidean', p=2, n_jobs=-1)
classifier.fit(X_train, y_train)
# Predicting the test set results
y_pred = classifier.predict(X_test)
# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
# Accuracy
TP = cm[0][0]
TN = cm[1][1]
FP = cm[1][0]
FN = cm[0][1]
accuracy = ((TP+TN)/(TP+TN+FP+FN))
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = (2*precision*recall)/(precision+recall)
print("Parameters Comprobation")
print("The accuracy of the model created is {:0.4f}.".format(accuracy))
print("The precision of the model created is {:0.4f}.".format(precision))
print("The recall of the model created is {:0.4f}.".format(recall))
print("The F1 score of the model created is {:0.4f}.\n".format(f1_score))

# Generating the data for exporting it into a .csv file
predictions = {'Survived':y_pred}
passengerid = {'PassengerId':test_dataset.iloc[:, 0].values}
predictions_dataset = pd.concat([pd.DataFrame(passengerid), pd.DataFrame(predictions)], axis=1)
predictions_dataset.to_csv('survived_predictions.csv', encoding='utf-8', index=False)