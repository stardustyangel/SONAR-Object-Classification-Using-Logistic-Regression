#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 07:18:17 2023

@author: tayssirboukrouba
"""

# import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# loading the data
df = pd.read_csv('sonar_data.csv', header=None)


# investigationg our data :
print(df.head())
print(f'This data has {df.shape[0]} rows and {df.shape[1]} cols')
print(df.describe())
print(df[60].value_counts())
print(df.groupby(60).mean())


# seperating dafarame into inputs and labels
X = df.drop(columns=60, axis=1)
y = df[60]


print(X.iloc[:3, :])
print(y)

# seperating data into train and test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, stratify=y, random_state=42)

# investigating our train and test inputs and labels
print(X.shape, X_train.shape, X_test.shape)

# Modelling
model = LogisticRegression()
model.fit(X_train, y_train)

# Train Evalutation
X_train_pred = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_pred, y_train)
print(f'Training data accuracy score :{round(training_data_accuracy*100,2)}')

# Test Evalutation
X_test_pred = model.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_pred, y_test)
print(f'Testing data accuracy score :{round(testing_data_accuracy*100,2)}')

# Testing our Predictive system
input_data = (0.0195, 0.0142, 0.0181, 0.0406, 0.0391, 0.0249, 0.0892, 0.0973, 
         0.0840, 0.1191, 0.1522, 0.1322, 0.1434, 0.1244, 0.0653, 0.0890, 0.1226
         ,0.1846, 0.3880, 0.3658, 0.2297, 0.2610, 0.4193, 0.5848, 0.5643,
         0.5448, 0.4772, 0.6897, 0.9797, 1.0000, 0.9546, 0.8835, 0.7662,
         0.6547, 0.5447, 0.4593, 0.4679, 0.1987, 0.0699, 0.1493, 0.1713,
         0.1654, 0.2600, 0.3846, 0.3754, 0.2414, 0.1077, 0.0224, 0.0155,
         0.0187, 0.0125, 0.0028, 0.0067, 0.0120, 0.0012, 0.0022, 0.0058,
         0.0042, 0.0067, 0.0012)

input_array = np.array(input_data) #shape = (60,)
reshaped_array = input_array.reshape(1,-1) #shape = (1,60)

prediction = model.predict(reshaped_array)

if prediction == 'M' : 
    print('The object is a Mine')
else : 
    print('The Object is a rock')

