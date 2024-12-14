#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 11:37:12 2024

@author: joemargolis
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import dataframe_image as dfi

X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv")
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv")

#Final Cleaning

def map_education_level(qualification):
    """
    Maps education qualifications to numeric categories:
    0: Class 12
    1: B* (Bachelor's degrees)
    2: M* or L* (Master's degrees and Law degrees)
    3: PhD
    
    Args:
        qualification (str): The education qualification
        
    Returns:
        int: The corresponding category number
    """
    # Convert to uppercase for consistent comparison
    qual = qualification.upper()
    
    if qual == 'CLASS 12':
        return 0
    elif qual == 'PHD':
        return 3
    elif qual.startswith('B'):
        return 1
    elif qual.startswith('M') or qual.startswith('L'):
        return 2
    else:
        raise ValueError(f"Unknown qualification: {qualification}")

le = LabelEncoder()
X_train['Gender'] = le.fit_transform(X_train['Gender'])
X_train['Working Professional or Student'] = le.fit_transform(X_train['Working Professional or Student'])
X_train['Gender'] = le.fit_transform(X_train['Gender'])
X_train['Sleep Duration'] = X_train['Sleep Duration'].map({'Less than 5 hours':0, '5-6 hours':1, '7-8 hours':2, 'More than 8 hours':3})
X_train['Dietary Habits'] = X_train['Dietary Habits'].map({'Unhealthy':0, 'Moderate':1, 'Healthy':2})
X_train['Degree'] = X_train['Degree'].apply(map_education_level)
X_train['Have you ever had suicidal thoughts ?'] = le.fit_transform(X_train['Have you ever had suicidal thoughts ?'])
X_train['Family History of Mental Illness'] = le.fit_transform(X_train['Family History of Mental Illness'])
X_train = X_train.drop('Name', axis=1)
X_train = X_train.drop('Profession', axis = 1) #Too Much dispersion, resulting in nearly an id/name type of variable


X_test['Gender'] = le.fit_transform(X_test['Gender'])
X_test['Working Professional or Student'] = le.fit_transform(X_test['Working Professional or Student'])
X_test['Gender'] = le.fit_transform(X_test['Gender'])
X_test['Sleep Duration'] = X_test['Sleep Duration'].map({'Less than 5 hours':0, '5-6 hours':1, '7-8 hours':2, 'More than 8 hours':3})
X_test['Dietary Habits'] = X_test['Dietary Habits'].map({'Unhealthy':0, 'Moderate':1, 'Healthy':2})
X_test['Degree'] = X_test['Degree'].apply(map_education_level)
X_test['Have you ever had suicidal thoughts ?'] = le.fit_transform(X_test['Have you ever had suicidal thoughts ?'])
X_test['Family History of Mental Illness'] = le.fit_transform(X_test['Family History of Mental Illness'])
X_test = X_test.drop('Name', axis=1)
X_test = X_test.drop('Profession', axis = 1)  #Too Much dispersion, resulting in nearly an id/name type of variable

# Instantiating sklear functions to be used
ros = RandomOverSampler(random_state=42)
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced')
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Lists to store scores
cv_accuracy_scores = []
best_accuracy = 0
best_model = None

#Running random forest model
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train), 1):
    # Split data
    X_train_fold = X_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_train_fold = y_train.iloc[train_idx]
    y_val_fold = y_train.iloc[val_idx]
    #Applying random oversampling to each fold training set
    X_train_fold, y_train_fold = ros.fit_resample(X_train_fold, y_train_fold)
    # Train model
    rf.fit(X_train_fold, y_train_fold.values.ravel())
    
    # Evaluate
    y_val_pred = rf.predict(X_val_fold)
    
    # Calculate accuracy
    fold_accuracy = accuracy_score(y_val_fold, y_val_pred)
    cv_accuracy_scores.append(fold_accuracy)
    
    # Store best model
    if fold_accuracy > best_accuracy:
        best_accuracy = fold_accuracy
        best_model = rf
    
    print(f"Fold {fold} Accuracy: {fold_accuracy:.4f}")
    
# Print cross-validation results
mean_cv_accuracy = np.mean(cv_accuracy_scores)
std_cv_accuracy = np.std(cv_accuracy_scores)
print(f"\nCross-validation results:")
print(f"Mean Accuracy: {mean_cv_accuracy:.4f} (+/- {std_cv_accuracy:.4f})")

# Evaluate on test set using best model
y_test_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
F1 = f1_score(y_test, y_test_pred, pos_label = 'Yes')
print(f"Test Set Accuracy: {test_accuracy:.4f}")
print(f"F1: {F1:.4f}")

cm = confusion_matrix(y_test, y_test_pred)
cm = pd.DataFrame(cm, index=['Truth: No Depression', 'Truth: Depression'], columns=['Predicted: No Depression', 'Predicted: Depression'])
dfi.export(cm, 'FullRFCM.png')
print(cm)
   

#Plotting Variable Importance

# list of column names from original data
cols = X_train.columns
# feature importances from random forest fit rf
rank = rf.feature_importances_
# form dictionary of feature ranks and features
features_dict = dict(zip(rank,cols))

plotDict = dict(sorted(features_dict.items(), key=lambda item: item[0], reverse=False))

parameters = list(plotDict.values())
importance = list(plotDict.keys())

plt.barh(parameters, importance)
plt.xlabel('Variable Importance')
plt.ylabel('Parameters')
plt.title('Depression Random Forest Variable Importance')
#plt.xticks(parameters, rotation=40, ha = 'right')
plt.savefig('RFFullImportance.png', bbox_inches='tight')
plt.show()



