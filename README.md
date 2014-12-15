datasciencePMLProject
=====================

Coursera DataScience Practical Machine Learning Project- Human Activity Recognition
# Machine Learning: Peer Assessment
## Human Activity Recognition:  Assessment of Activity Quality by Machine Learning Techniques
---
---
### James C. Weaver

## Introduction
Traditionally, research on human activity recognition (HAR) has focused on identifying and predicting which activity was performed at some specific time.  In their conference paper at the 2013 Augmented Human International Conference, Velloso and Bulling et al. extended this concept to the identification of how well an activity was performed by the wearer of the activity recognition device (1).  To do this, they asked six male participants with little weight lifting experience and between the ages of 20 and 28 to perform 10 repetitions of the activity “Unilateral Dumbbell Biceps Curl” in five different fashions, Classes A through E.  Class A corresponds to the correct specification of the activity; classes B through E correspond to common mistakes made by people performing the exercise.  Participants wore sensors during the exercise which captured three-axes acceleration, gyroscope and magnetometer data.  Each participant had four such sensors mounted in the user’s glove, armband, lumbar belt and the dumbbell itself.  From these data, 96 feature sets were generated for each sensor.  In the dataset analyzed in this course project, 38 feature variables were made available for each sensor.

In their conference paper, Velloso and Bulling used machine learning techniques to detect mistakes by classification.  They used a Random Forest approach “because of the characteristic noise in the sensor data,” achieving an overall recognition performance of 98.03%.  In this course project, we also analyze a subset of the Velloso and Bulling data with a random forest model, and assess prediction accuracy with their overall accuracy recognition performance as our benchmark measure.

## Model Building
### Data acquisition and cleaning
Training and testing data were down loaded from the following websites and stored in the working directory for R:
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The testing data set was loaded into R and was found to consist of 20 cases and 160 variables.  The first 7 variables dealing with the user and data acquisition process are not useful in the data analysis and were removed from the dataset.  Many of the 152 feature variables contained missing or NA data.  If these missing data were a small percentage of the feature variable, they could be replaced by imputing values from the remaining data.  Thus, we choose to remove those feature variables from the dataset which had more than 75% missing values.  This resulted in retaining 52 feature variables in the dataset, none of which contained any missing values.  Therefore, data imputation was not necessary.

The training data were loaded into R and found to consist of 19622 cases and 160 variables.  The same criteria used for the testing data was applied to the training data, and the same 52 feature variables were retained.

### Data Partitioning
The testing data were used to evaluate out of sample performance of our prediction model by predicting the Class for each of the 20 cases.  The training data was randomly partitioned into two sets, a training set consisting of 60% of the cases and a validation set consisting of 40% of the cases.  The training set was used to build the prediction model.  The validation set was used to evaluate out of sample performance of the model.

### Random Forest Model
A random forest prediction model was built using the caret package in R.  The model was trained using 5 fold cross validation so that an expected sample error rate could be determined and then compared with the performance of the model on the validation set.  Since the time required to train the random forest model is quit long (about 1 hour), the resulting model, rfModelcv, was saved to the working directory for subsequent model evaluation and predictions.  Note also that the importance attribute was set to TRUE in the train statement so that feature rank could be assessed.

## Model Evaluation
### Model Output
The random forest model with 5 fold cross validation resulted in a final model with an overall accuracy of 0.9887 at an mtry value of 27.  The out of bag (OOB) estimate of error rate was 0.82%.

### Model Evaluation on the Validation Data Partition
When evaluated on the validation data partition set, the final model produced and overall accuracy of 0.9916 (95% CI of 0.9893, 0.9935).  This corresponds to an error rate of 0.84% which agrees well with the OOB estimate of 0.82% from cross validation and the 1.97% error rate reported by Velloso and Bulling et al. on the dataset with the larger feature set.

### Class predictions for the testing data set, “testingData”
The final random forest model was then applied to the testing dataset to predict the class for each of the 20 cases contained in that set.  The model achieved 100% accuracy on this set as indicated by the Coursera submission.

### Feature Rank by Importance
Feature rank by importance with respect to model accuracy was estimated with the varImp() function.  The 20 most important variables (out of the 52 features) are given below.

### Feature Selection using Recursive Feature Elimination (RFE Method in caret)
Using the recursive feature elimination function in caret, all possible subsets of the predictor variables are explored in order to identify that subset of features responsible for most determining model accuracy.  As seen below, five features are responsible for 95.6% of the prediction accuracy:  roll_belt, yaw_belt, magnet_dumbbell_z, pitch_belt, magnet_dumbbell_y.  Further note that this result is consistent with the importance ranking described above.  The five features identified by RFE are in the set of the six most important features identified by the varImp() function.

In their work, Velloso and Bulling identified 17 features as the most important for prediction:  7 in the belt, 3 in the arm, 3 in the glove and 4 in the dumbbell (1).  In our work, features in only the belt and dumbbell were responsible for over 95% of the prediction accuracy.  Figure 1 shows a plot of cumulative accuracy as a function of the ranked predictors from the RFE analysis.  Note the break in the curve after the 6th predictor.  The six most important predictors account for 97.27% of the final model accuracy.

#### Figure 1:  Cumulative Accuracy as a function of the Ranked Predictors from the RFE Analysis

It is also interesting to compare this set of features with the set of highly correlated variables in the training set.  The first five predictors identified by the RFE analysis are fully contained within the set of highly correlated variables in the training partition of the trainingData set.  Thus, it would have been ill-advised to have removed the highly correlated variables from the training set in advance of building the random forest model.

## Conclusions
A random forest machine learning algorithmic model was built from the human activity recognition dataset of Velloso and Bulling et al. and applied to predict the class of activity in the Unilateral Dumbbell Biceps Curl exercise.  The model has an out of sample prediction accuracy of 99.16% (0.9893, 0.9935) on a truncated feature set of the Velloso-Bulling data, a value which compares well with their reported accuracy using the random forest algorithm of 98.03%.  A six member subset of the features responsible for over 97% of the model accuracy was also identified using Recursive Feature Elimination.  This subset consisted of features from sensors located primarily in the belt and dumbbell.

## Reference
1)	E. Velloso, A. Bulling, H. Gellersen, W. Ugulino and H. Fuks, Qualitative Activity Recognition of Weight Lifting Exercises, Augmented Human International Conference (AH), March 2013. Stuttgart, Germany: ACMSIGCHI, 2013.
