# Data-Mining-Final

## Data Cleaning and Preprocessing

The dataset for this project is called Depression Survey/Dataset for Analysis. It was put together by an anonymous research group and posted by Suman Sharma. It contains the depression survey information for 2,556 patients from all over India, with ages ranging from 18-60 years old. This dataset focuses mainly on lifestyle factors like `Dietary Habits` or `Working Professional or Student` to find connections to the target variable `Depression` which is a binary output, either diagnosed with depression and not.

The main cleaning for this code can be found in `Cleaning_Code.py`. To run this file, you must have the kaggle dataset saved in the same directory as this file with the name "final_depression_dataset_1.csv". You must also install the following packages:

* numpy
* pandas
* matplotlib
* sklearn



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
This project uses a Naive Bayes Classifier to predict depression based on various features like age, profession, sleep duration, and stress levels. We implemented Gaussian Naive Bayes for numerical data and performed a step-by-step analysis to improve the model's performance.

Files in the Project
final_project.py: The Python script that runs the model and produces the results.
X_train.csv: Training data features.
y_train.csv: Training data labels.
X_test.csv: Testing data features.
y_test.csv: Testing data labels.
README.md: Instructions for running and understanding the project.
Visual Outputs: Confusion matrix, ROC curve, feature importance, and correlation matrix graphs.
Setup Instructions
Install Dependencies
Ensure the following libraries are installed on your system:

bash
Copy code
pip install pandas scikit-learn matplotlib seaborn
Run the Script
Place all the required files (X_train.csv, y_train.csv, X_test.csv, y_test.csv) in the same directory as final_project.py. Run the script:

bash
Copy code
python final_project.py
Output
The script will output the following:

Top 15 important features using Mutual Information.
A correlation matrix heatmap for the top features.
Model performance metrics (Accuracy, Precision, Recall, F1-Score).
A Confusion Matrix to show correct and incorrect predictions.
An ROC Curve with the AUC score to measure overall performance.
A Learning Curve to observe model behavior with different amounts of data.
Steps in the Project
Feature Selection

Why it’s done: To reduce unnecessary features and improve model accuracy.
How: Mutual Information was used to identify the 15 most important features.
Feature Correlation Analysis

A heatmap was created to check correlations between features.
This step confirms that features are not strongly correlated, which is good for Naive Bayes (assumes feature independence).
Model Training

Gaussian Naive Bayes was used, which assumes data follows a normal (Gaussian) distribution.
Model Evaluation

Confusion Matrix: Shows True Positives, True Negatives, False Positives, and False Negatives.
ROC Curve and AUC: Measures how well the model separates classes.
Learning Curve: Helps us understand how the model performs as we increase the training data size.
Model Results
Confusion Matrix:

Shows that the model correctly predicts most "No Depression" cases but struggles with False Positives (predicting depression when it’s not).
Precision and Recall:

Precision for No Depression: 99%
Recall for Depression Detected: 96%
Observation: The model is better at identifying “No Depression” cases due to class imbalance.
ROC Curve and AUC:

AUC Score = 0.90, which means the model performs well in separating the two classes.
Learning Curve:

Shows how the model improves with more data. It highlights that increasing training data helps stabilize the accuracy.
Observations
Age and Profession (Student/Working) were the most significant features for predicting depression.
The model performs well overall but struggles with False Positives, likely due to fewer examples of depression in the dataset.
The AUC score of 0.90 confirms that the model can effectively separate “Depression” and “No Depression” classes.
Future Improvements
Handle class imbalance by oversampling techniques like SMOTE.
Use additional models (e.g., Logistic Regression, Random Forest) for comparison.
Perform hyperparameter tuning to optimize the Naive Bayes model.
Conclusion
This project demonstrates how a Naive Bayes Classifier can predict depression based on key features. The process included feature selection, visualization, model training, and evaluation. By focusing on the most important features and analyzing results through graphs and metrics, we identified the strengths and limitations of the model.

## Random Forest Full Model

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
