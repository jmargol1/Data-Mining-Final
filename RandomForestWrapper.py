#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 13:21:52 2024

@author: joemargolis
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import dataframe_image as dfi
from sklearn.feature_selection import SequentialFeatureSelector
from copy import deepcopy

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

#Instantiating sklearn functions to be used 
ros = RandomOverSampler(random_state=42)
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced')
kf = KFold(n_splits=10, shuffle=True, random_state=42)

#Creating Lists to store outputs, beginning with values from full model
n_features = len(X_train.columns)
eliminatedList = ["None"]
trainAccuracy = [0.9550]
testAccuracyList = [95.31]
testF1List = [84.85]
numRemaining = [12]
remaining_features = X_train.columns.tolist()

# Store best overall model
best_overall_accuracy = 0
best_overall_model = deepcopy(rf)
best_feature_set = remaining_features.copy()

#Running Wrapper Method
for i in range(n_features - 1):
    # Fit SFS
    sfs = SequentialFeatureSelector(
        estimator=rf,
        n_features_to_select=n_features - i - 1,
        direction = 'backward'
    )
    
    sfs.fit(X_train[remaining_features], y_train.values.ravel())
    
    # Get eliminated feature
    eliminated = [f for f, s in zip(remaining_features, sfs.support_) if not s][0]
    remaining_features = [f for f in remaining_features if f != eliminated]
    
    # Cross validation
    cv_accuracy_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train), 1):
        # Split data
        X_train_fold = X_train.iloc[train_idx][remaining_features]
        X_val_fold = X_train.iloc[val_idx][remaining_features]
        y_train_fold = y_train.iloc[train_idx]
        y_val_fold = y_train.iloc[val_idx]
        # Random oversampling for each fold
        X_train_fold, y_train_fold = ros.fit_resample(X_train_fold, y_train_fold)
        # Train model
        rf.fit(X_train_fold, y_train_fold.values.ravel())
        
        # Evaluate
        y_val_pred = rf.predict(X_val_fold)
        
        # Calculate accuracy
        fold_accuracy = accuracy_score(y_val_fold, y_val_pred)
        cv_accuracy_scores.append(fold_accuracy)
    
    mean_cv_accuracy = np.mean(cv_accuracy_scores)    
    trainAccuracy.append(mean_cv_accuracy)
    # Store best model
    if mean_cv_accuracy > best_overall_accuracy:
        best_accuracy = fold_accuracy
        best_model = rf
        
     # Evaluate on test set
    rf.fit(X_train[remaining_features], y_train.values.ravel())
    test_accuracy = accuracy_score(y_test, rf.predict(X_test[remaining_features]))
    F1 = f1_score(y_test, rf.predict(X_test[remaining_features]), pos_label = 'Yes')
    testAccuracyList.append(round(test_accuracy * 100, 2 ))
    testF1List.append(round(F1 * 100, 2))
    eliminatedList.append(eliminated)
    
    #Printing Results at Each Step
    print(f"\nStep {i+1}:")
    print(f"Eliminated feature: {eliminated}")
    print(f"Number of features remaining: {len(remaining_features)}")
    print(f"Mean CV Accuracy: {mean_cv_accuracy:.4f} (±{np.std(cv_accuracy_scores):.4f})")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test F1: {F1:.4f}")
    
# Plotting Test Accuracies with varying number of parameters
plotAccuracy = testAccuracyList[::-1]
plotF1 = testF1List[::-1]
x = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
plt.plot(x, plotAccuracy, label = 'Test Accuracy')
plt.plot(x, plotF1, label='Test F1')
plt.legend()
plt.xlabel('Number of Parameters')
plt.ylabel('Metric Output (%)')
plt.title('Test Accuracy and F1 Score By Number of Parameters')
plt.savefig('WrapperOutputs.png')
plt.show()

#Creating Table with the model success metrics at each step 
wrapperTable =  np.column_stack((eliminatedList, x[::-1], testAccuracyList, testF1List))
agl_row = np.array(["Age", 0, "-", "-"])
wrapperTable = np.vstack([wrapperTable, agl_row])
wrapperTable = pd.DataFrame(wrapperTable, columns=['Eliminated Feature', '# Features Remaining', 'Test Accuracy (%)','Test F1-Score (%)'])
dfi.export(wrapperTable, 'WrapperImportance.png')

