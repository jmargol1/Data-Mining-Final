# Data-Mining-Final

## Introduction

The prevalence of depression as a leading mental health challenge underscores the need for innovative and effective approaches to early detection. Informed by decades of research, studies have shown that demographic factors, lifestyle choices, and environmental stressors play critical roles in influencing mental health outcomes. Our work seeks to build upon this foundation by leveraging predictive modeling techniques to assess depression risk factors. This study utilizes a comprehensive dataset encompassing variables such as age, academic and work pressures, and lifestyle habits, collected from over 2,500 individuals in India.

The comprehensive report on this topic, outlining how we built our models and analysis of our results can be found in the `Project_Write_Up.docx` file. Our concise presentation can be found in the `Depression_Slides.docx` file.

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

This portion uses a Naive Bayes Classifier to predict depression based on various features like age, profession, sleep duration, and stress levels. We implemented Gaussian Naive Bayes for numerical data and performed a step-by-step analysis to improve the model's performance.

The code used to build and run our Naive-Bayes Madels can be found in the file `final_project.py`. To run this code, you must first run the `Cleaning_Code.py` file, and ensure that the datasets produced from that file are all saved in the same directory as `final_project.py`. This includes the following files:

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

The code used to build and run our K-Nearest Neighbors models can be found in the file `KNN.ipynb`. To run this code, you must first ensure that the following files are saved in the same directory as `KNN.ipynb`. 

* "X_train_resamp.csv"
* "y_train_resamp.csv"

You must also first run the `Cleaning_Code.py` file, and ensure that the datasets produced from that file are all saved in the same directory as `KNN.ipynb`. This includes the following files:

* "X_test.csv"
* "y_test.csv"

You must also install the following additional packages:

* imb_learn
* seaborn
* numpy
* pandas
* matplotlib
* scikit_klearn

Running this code will produce the following outputs:

* Feature Weights for KNN Model
* Classification report for KNN Model results for model with all parameters
* Correlation matrix between features for model with all parameters
* Plot showing training and testing accuracy vs. number of parameters for wrapper method models
* Classification report for KNN Model results for the wrapper method selected model
* Classification report for gridsearchcv KNN Model

## Random Forest Full Model

The code used to build and run the random forest model with all 12 parameters can be found in the file `RandomForest.py`. To run this code, you must first run the `Cleaning_Code.py` file, and ensure that the datasets produced from that file are all saved in the same directory as `RandomForest.py`. This includes the following files:

* "X_train.csv" (The input parameters for the training dataset)
* "y_train.csv" (The depression classifications for the training dataset)
* "X_test.csv" (The input parameters for the testing dataset)
* "y_test.csv" (The depression classifications for the testing dataset)

You must also install the following package:

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

The code used to build and run the backwards feature selection wrapper method for our random forest model can be found in the file `RandomForestWrapper.py`. To run this code, you must first run the `Cleaning_Code.py` file, and ensure that the datasets produced from that file are all saved in the same directory as `RandomForestWrapper.py`. This includes the following files:

* "X_train.csv" (The input parameters for the training dataset)
* "y_train.csv" (The depression classifications for the training dataset)
* "X_test.csv" (The input parameters for the testing dataset)
* "y_test.csv" (The depression classifications for the testing dataset)

You must also install the following additional packages:

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
