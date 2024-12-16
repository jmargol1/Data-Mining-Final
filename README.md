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
