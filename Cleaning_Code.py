#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 12:09:17 2024

@author: joemargolis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv("final_depression_dataset_1.csv")


# Merging Academic and Work Pressure into One Column
df['Academic/Work Pressure'] = np.where(
    pd.notnull(df['Work Pressure']),
    df['Work Pressure'],
    df['Academic Pressure']
    )
# Checking The Merge Did not delete values
total_value_check = (pd.notnull(['Work Pressure']) | pd.notnull(df['Academic Pressure'])).sum()
merged_value_check = pd.notnull(df['Academic/Work Pressure']).sum()
# Dropping The Work and Academic Pressure Columns
df.drop(['Work Pressure', 'Academic Pressure'], axis=1, inplace=True)

# Merging Academic and Work Satisfaction into One Column
df['Academic/Work Satisfaction'] = np.where(
    pd.notnull(df['Job Satisfaction']),
    df['Job Satisfaction'],
    df['Study Satisfaction']
    )
# Checking The Merge Did not delete values
total_value_check = (pd.notnull(['Job Satisfaction']) | pd.notnull(df['Study Satisfaction'])).sum()
merged_value_check = pd.notnull(df['Academic/Work Satisfaction']).sum()
# Dropping The Work and Academic Pressure Columns
df.drop(['Job Satisfaction', 'Study Satisfaction'], axis=1, inplace=True)

# REmoving CGPA Column as it only relates to the students
df.drop(['CGPA'], axis=1, inplace=True)
df.drop(['City'], axis =1, inplace = True)

# Checking Imbalance

imbalance = df.groupby('Depression')['Depression'].value_counts()

imbalance.plot(kind='bar')
plt.xticks([0,1], ['No', 'Yes'])
plt.title('Value Counts of Depression')
plt.xlabel('Depression (Yes/No)')
plt.ylabel('Count')
plt.savefig('imbalance.png', bbox_inches='tight')
plt.show()


grouped = df.groupby(['Working Professional or Student', 'Profession'], dropna = False).size().reset_index(name='count')

# Fixing nan values in the Profession Column
null_mask = df['Profession'].isnull()
is_student = df['Working Professional or Student'].str.lower().str.contains('student', na=False)
df.loc[null_mask & is_student, 'Profession'] = 'student'
df.loc[null_mask & ~is_student, 'Profession'] = 'unemployed'

#Plotting Depression Probability by Age Group
# Convert Depression to numeric
# If your Depression column has 'Yes'/'No' or 'True'/'False', use this line:
AgeData = df.copy()
AgeData['Depression'] = pd.to_numeric(AgeData['Depression'].map({'Yes': 1, 'No': 0}))
# If your Depression column just needs to be converted to numeric, use this line instead:
# data['Depression'] = pd.to_numeric(data['Depression'])

# Create age groups with 10-year bins
bins = np.arange(18, AgeData['Age'].max() + 6, 6)
AgeData['AgeGroup'] = pd.cut(AgeData['Age'], bins=bins, right=False)

# Calculate depression percentage for each age group
depression_by_age = AgeData.groupby('AgeGroup').agg({
    'Depression': ['count', 'sum']
}).reset_index()

depression_by_age.columns = ['AgeGroup', 'Total', 'Depressed']
depression_by_age['Percentage'] = (depression_by_age['Depressed'] / 
                                 depression_by_age['Total'] * 100)

# Create the plot
plt.figure(figsize=(18, 6))
plt.bar(range(len(depression_by_age)), depression_by_age['Percentage'])

plt.title('Depression Rate by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Percentage with Depression (%)')

plt.xticks(range(len(depression_by_age)), 
          [f"{int(group.left)}-{int(group.right-1)}" 
           for group in depression_by_age['AgeGroup']])
plt.savefig('DepressionByAge.png', bbox_inches='tight')
plt.show()


# Creating Train/Test Splits

# Split the data into features (X) and target variable (y)
X = df.drop('Depression', axis=1)
y = df['Depression']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#Saving Cleaned Training and Test Sets
X_train.to_csv('X_train.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_test.to_csv('y_test.csv', index=False)



