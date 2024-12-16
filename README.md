# Data-Mining-Final

## Data Cleaning and Preprocessing

The dataset for this project is called Depression Survey/Dataset for Analysis. It was put together by an anonymous research group and posted by Suman Sharma. It contains the depression survey information for 2,556 patients from all over India, with ages ranging from 18-60 years old. This dataset focuses mainly on lifestyle factors like `Dietary Habits` or `Working Professional or Student` to find connections to the target variable `Depression` which is a binary output, either diagnosed with depression and not.

The main cleaning for this code can be found in `Cleaning_Code.py`. To run this file, you must have the kaggle dataset saved in the same directory as this file with the name "final_depression_dataset_1.csv". You must also install the following packages:

* numpy
* pandas
* matplotlib
* scikit_klearn



Running this file produces the initially cleaned datasets to be used in the models, and saves these datasets to your directory.

* "X_train.csv" (The input parameters for the training dataset)
* "y_train.csv" (The depression classifications for the training dataset)
* "X_test.csv" (The input parameters for the testing dataset)
* "y_test.csv" (The depression classifications for the testing dataset)

It will also produce and save the following bar plots to the same directory:

* "imbalance.png" --> A bar plot showing the imbalance between depression and no depression
* "DepressionByAge.png" --> A bar plot displaying the proportion of age groups that were diagnosed with depression

## Naive Bayes Model

Naive Bayes Classifier for Predicting Depression
Project Overview
This portion uses a Naive Bayes Classifier to predict depression based on various features like age, profession, sleep duration, and stress levels. We implemented Gaussian Naive Bayes for numerical data and performed a step-by-step analysis to improve the model's performance.

The code used to build and run the backwards feature selection wrapper method for our random forest model can be found in the file `final_project.py`. To run this code, you must first run the `Cleaning_Code.py` file, and ensure that the datasets produced from that file are all saved in the same directory as `final_project.py`. This includes the following files:

* "X_train.csv" (The input parameters for the training dataset)
* "y_train.csv" (The depression classifications for the training dataset)
* "X_test.csv" (The input parameters for the testing dataset)
* "y_test.csv" (The depression classifications for the testing dataset)

You must also install the following additional package:

* seaborn
  
Visual Outputs: Confusion matrix, ROC curve, feature importance, and correlation matrix graphs.



Running the script will output the following:

* Top 15 important features using Mutual Information.
* A correlation matrix heatmap for the top features.
* Model performance metrics (Accuracy, Precision, Recall, F1-Score).
* A Confusion Matrix to show correct and incorrect predictions.
* An ROC Curve with the AUC score to measure overall performance.
* A Learning Curve to observe model behavior with different amounts of data.

## KNN Model

#Using KNN to predict depression 

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
# load cleaned data
x_train=pd.read_csv('/Users/andrechuabio/Library/Mobile Documents/com~apple~CloudDocs/Fall24/Data mining/cleaneddata (1)/X_train_resamp.csv')
y_train=pd.read_csv('/Users/andrechuabio/Library/Mobile Documents/com~apple~CloudDocs/Fall24/Data mining/cleaneddata (1)/Y_train.csv')
x_test=pd.read_csv('/Users/andrechuabio/Library/Mobile Documents/com~apple~CloudDocs/Fall24/Data mining/cleaneddata (1)/X_test_resamp.csv')
y_test=pd.read_csv('/Users/andrechuabio/Library/Mobile Documents/com~apple~CloudDocs/Fall24/Data mining/cleaneddata (1)/Y_test.csv')

```

```python
y_train = y_train.squeeze()
y_test = y_test.squeeze()
print("shape of x_train",x_train.shape)
print("shape of y_train",y_train.shape)
print("shape of x_test",x_test.shape)
print("shape of x_test",x_test.shape)

x_test=x_test.iloc[:len(y_test)]
y_train=y_train.squeeze()
y_test=y_test.squeeze()

label_encoder=LabelEncoder()
y_train=label_encoder.fit_transform(y_train)
y_test=label_encoder.transform(y_test)

label_encoders={}
feature_mappings={}
for col in x_train.select_dtypes(include=['object']):
    le=LabelEncoder()
    x_train[col]=le.fit_transform(x_train[col])
    x_test[col]=le.transform(x_test[col])
    label_encoders[col]=le
    feature_mappings[col]=dict(zip(le.classes_,le.transform(le.classes_)))

for feature, mapping in feature_mappings.items():
    print(f"Encoding for {feature}:{mapping}")

weights=x_train.corrwith(pd.Series(y_train)).abs()
weights=weights/weights.sum()

print("feature weights:\n",weights)

X_train_weighted=x_train*weights
X_test_weighted=x_test*weights

if x_test.shape[0]> y_test.shape[0]:
    x_test=x_test[:y_test.shape[0]]
elif y_test.shape[0]>x_test.shape[0]:
    y_test=y_test[:x_test.shape[0]]

scaler=StandardScaler()
X_train_weighted=scaler.fit_transform(X_train_weighted)
X_test_weighted=scaler.transform(X_test_weighted)

knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_weighted,y_train)
y_pred=knn.predict(X_test_weighted)

accuracy=accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)
print("classification report:\n",classification_report(y_test,y_pred))
```
To start off, we decided to include all features as a base test before deciding which ones to remove. We began by verifying the shape of the input data sets to ensure consistency. We also made sure to convert the y train and test to 1D arrays for compatibility. We then proceeded to label encoding to the categorical target variable (y train and test) to convert the "Yes/no" labels into numeric values (0s and 1s). We also used the Label encoder for all columns with object data types to encode the categorical features in x train and test. The mappings for encoded values were stored for reference. We then looked at correlation -based weights for each feature by computing the absolute correlation between x_train features and the target y_train. The weights were normalized to sum to 1 and were printed for verification and reference. To apply the weights effectively, we multiplied the x train and test datasets element wise by the calculated feature weights in order to scale the features based on their importance

Next, we normalized the data using StandardScaler to standardize features to have zero mean and unit variance. This ensures that the KNN algo (which is sensitive to the scale of data) performs optimally. We decided to initialize the KNN algorithm with 5 neighbours and trained it with the weighted and normalized x train and y train data. Predictions were made on the x test data.

From the initial classification report and weights, we found that the model achieved an accuracy of 85.15% with a strong performance on the majority class (no for depression) but found to struggle with the minority class (yes for depression) for 'no depression", precision was 0.97 and recall was 0.85, resulting in a f-1 score of 0.91. In contrast, class 1 (yes for depression) had a lower precision (0.53) but a higher recall (0.88) leading to a F-1 score of 0.66. The most influential features were Age (0.2), working professional or student (0.14), and profession(0.11) while features like name and family history of mental illness contributed minimally. The results led us to address this class imbalance (429 vs 83 samples) through oversampling or class weighting or improving feature selection. It also tells us that it performed well for predicting no depression but needs improvement on predicting a person has depression. 


```python
# check on cross corr between features to see if any are redundant
correlation_matrix=x_train.corr()

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix,annot=True,fmt='.2f',cmap="coolwarm")
plt.title("Feature corelation matrix")
plt.show
```
After our initial attempt we decided to carry out feature selection to see if any improvement can be made. We first proceeded with making a feature correlation matrix to see if any features are redundant. From the matrix we found that age and working professional or student have a moderate correlation (0.49) indicating that older individuals are more likely to be working professionals. Profession showed a strong negative correlation with working professional or student (-0.57), suggesting it strongly distinguishes between students and professionals which is logical. Age also had a moderate negative correlation with profession (-0.42), reflecting the differing roles based on age. The other features like sleep, diet and family history of mental illness exhibited negligible correlations with other variables, while name shows no meaningful relationship. After gaining these insights we decided to drop the name feature and do further checks on each features impact on model accuracy if dropped.

```python
#wrapper ( sequential forward selection)
selected_feat=[]
scores_each_step=[]
remaining_feat=list(x_train.columns)
best_score=0

while remaining_feat:
    scores=[]
    for feature in remaining_feat:
        current_feat=selected_feat+[feature] # subset of features to evaluate
        x_train_sub=x_train[current_feat]
        x_test_sub=x_test[current_feat]

        knn=KNeighborsClassifier(n_neighbors=5)
        knn.fit(x_train_sub,y_train)
        y_pred=knn.predict(x_test_sub)
        score=accuracy_score(y_test,y_pred)
        scores.append((feature,score))

    scores.sort(key=lambda x:x[1],reverse=True)
    best_feature,best_feature_score=scores[0]

    if best_feature_score>best_score:
        selected_feat.append(best_feature) #add best feature
        remaining_feat.remove(best_feature) #remove it from remaining features
        best_score-=best_feature_score  #update best score
        scores_each_step.append(best_feature_score)
        print(f"Selected feature :{best_feature},score:{best_feature_score}")
    else:
        break    # stop if no improvement

print(f"Final selected features:{selected_feat}")
import matplotlib.pyplot as plt

#plot feature selection results
plt.figure(figsize=(10,6))
plt.plot(range(1, len(scores_each_step)+1),scores_each_step,marker='o',linestyle='-',color='b')
plt.xticks(range(1,len(selected_feat)+1),selected_feat,rotation=45,ha='right')
plt.xlabel('Selected Features in order')
plt.ylabel('accuracy score')
plt.title('Wrapper Method: Accuracy vs selected features')

plt.grid()
plt.show()
```
From the wrapper approach we found that the best features to include are Age, 'Have you ever had suicidal thoughts?' and 'Working Professional or student' with an improved accuracy score of 0.8906 (up from 0.85) we tried both sequential forward selection and backward and found forward selection yielded the best insights 

```python
# wrapper continued with best features

selected_features=['Age', 'Have you ever had suicidal thoughts ?', 'Working Professional or Student']
scores_at_each_step=[]
X_train_filtered=x_train[selected_features]
X_test_filtered=x_test[selected_features]

knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_filtered,y_train)
y_pred=knn.predict(X_test_filtered)

accuracy=accuracy_score(y_test,y_pred)
classification_rep=classification_report(y_test,y_pred)
print("accuracy:",accuracy)
print("classification_report:\n",classification_rep)
```

From the wrapper approach we found that the best features to include are Age, 'Have you ever had suicidal thoughts?' and 'Working Professional or student' with an improved accuracy score of 0.8906 (up from 0.85) we tried both sequential forward selection and backward and found forward selection yielded the best insights 

From the classification report we can see a significant improvement after implementing feature selection (wrapper method). Initially the model achieved 85% accuracy, with class 0 (no) having a precision of 0.97 and recall of 0.85, while class 1 (yes) had a precision of 0.53 and a recall of 0.88. Post wrapper feature selection, the accuracy increased to 88.67, with improved balance, class 1 precision rose to 0.64 (from 0.53), while its recall dropped slightly to 0.67. Class 0 maintained strong precision and recall which contributed to overall improvement. The wrapper method successfully optimized feature selection, enhancing accuracy and precision for the minority class without significantly compromising recall.

Next, we tried using the grid search cross validation approach, which optimized hyperparameters like the number of neighbours, distance weighting and distance metric. This resulted in a accuracy of 82.23%. While this method identified the best parameters (neighbours =3, weights ='distance, metric ='Manhattan), the overall accuracy and performance was lower compared to the wrapper-based feature selection approach. Specifically, class 1 performance dropped further, with precision at 0.47 and a f1-score of 0.59, despite a recall of 0.78. The reduced effectiveness could be attributed to the inclusion of less informative features as grid search CV tunes hyperparameters but does not inherently perform feature selection. In contrast, the wrapper method focused on selecting the most impactful features which led to a more streamlined model with better accuracy and balance across both classes. This was useful knowing which hyperparameters are useful but not much for feature selection. Going forward we made sure the right hyperparameters (e.g. k=3) are used from this insight.


```python
from sklearn.model_selection import GridSearchCV


x_train_filtered = x_train.drop(columns=['Name'])
x_test_filtered = x_test.drop(columns=['Name'])

knn=KNeighborsClassifier()
param_grid={ 'n_neighbors':[3,5,7,9],
            'weights':['uniform','distance'],
            'metric':['euclidean','manhattan']}

grid_search=GridSearchCV(estimator=knn,param_grid=param_grid,cv=5,scoring='accuracy')

grid_search.fit(x_train_filtered, y_train)
best_knn = grid_search.best_estimator_
y_pred = best_knn.predict(x_test_filtered)

print("Best parameters",grid_search.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
```


Finally, we tried out the SMOTE technique (synthetic minority oversampling technique) to deal with a class imbalance issue. As we had 429 samples for no depression against 83 samples for yes to depression. This imbalance causes the model to favour class 0 (no), which led to a low precision for class 1. Since KNN is a distance-based algorithm, the imbalanced data set makes it harder to classify minority class instances correctly since the majority class dominates the neighbourhood. Post -SMOTE, we found that it helped balancing the classes and improving precision for class 1, however, it resulted in an overall accuracy drop due to the inclusion of synthetic data, which may introduce noise.


```python
# TRY out smote 

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from collections import Counter

print('class dist before SMOTE:',Counter(y_train))

smote=SMOTE(random_state=42)
x_train_resampled,y_train_resampled=smote.fit_resample(x_train,y_train) #apply smote

print("class distribution after smote:",Counter(y_train_resampled))

knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train_resampled,y_train_resampled) # train model with resampled data

y_pred=knn.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)

print("accuracy",accuracy)
print("classification report:\n",classification_rep)

```

In conclusion, the KNN model was optimized using smote, wrapper-based feature selection, and gridsearchCV to address class imbalance and improve performance. The baseline model achieved 85.15 accuracy, with poor precision for class 1 (0.53). applying SMOTE balanced the classes, improving class 1 precision to 0.64, but overall accuracy dropped to 75.39% due to noise from synthetic data. Wrapper based feature selection identified key features (age, suicidal thoughts, working professional or student) and achieved the best accuracy of 88.67% significantly improving model performance. in comparison, gridsearchcv optimized hyperparameters but achieved a lower accuracy of 82.23%. Overall, combining wrapper-based feature selection with optimized hyperparameters yields the best results, while integrating SMOTE can further enhance class 1 (yes to depression) performance.
## Random Forest Full Model
When utilizing Random forests

The code used to build and run the random forest model with all 12 parameters can be found in the file `RandomForest.py`. To run this code, you must first run the `Cleaning_Code.py` file, and ensure that the datasets produced from that file are all saved in the same directory as `RandomForest.py`. You must also install the following package:

* dataframe_image

Running this file will print in the console the following metrics:

* CV Fold Accuracy for each of the ten folds
* Mean accuracy of the ten CV Fold accuracies
* Test Set Accuracy of the final model
* Test F1-Score of the final model
* Final Model Confusion Matrix

Running this file will also produce and save the following images to the same directory:

* "FullRFCM.png" --> Final Model Confusion Matrix
* "RFFullImportance.png" --> Bar plot displaying the variable importance for each parameter

## Random Forest Backwards Feature Selection

The code used to build and run the backwards feature selection wrapper method for our random forest model can be found in the file `RondomForest.py`. To run this code, you must first run the `Cleaning_Code.py` file, and ensure that the datasets produced from that file are all saved in the same directory as `RandomForest.py`. You must also install the following additional packages:

* dataframe_image
* copy

Running this file will print the following in the console:

* Model metrics for each step of feature elimination
  * Number of Features Remaining
  * Mean CV Fold Accuracy
  * Test Accuracy
  * Test F1-Score
 
Running this file will also produce and save the following images to the same directory:

* "WrapperOutputs.png" --> A line plot showing the movement of Test Accuracy and Test F1-Score at each number of parameters remaining
* "WrapperImportance.png" --> A table outlining the feature eliminated at each step, along with the number of features remaining, and the Test Accuracy and F1-Score for each model.
