# Practical Machine Learning Course Project 


Load the required packages for running the random forest model in caret.


```r
## Load the required packages.

library (lattice)
library (ggplot2)
library (caret)
```

```
## Warning: package 'caret' was built under R version 3.2.5
```

```r
library (randomForest)
```

```
## Warning: package 'randomForest' was built under R version 3.2.5
```

```
## randomForest 4.6-12
## Type rfNews() to see new features/changes/bug fixes.
```

Read the training and testing files.


```r
## Read the training and testing files.

setwd ('C:/Users/Steve/Desktop')
exercise_training <- read.csv ("pml-training.csv")
exercise_testing <- read.csv ("pml-testing.csv")
```
An inspection of the training and testing files reveals that the first seven 
columns (blank, user_name, raw_timestamp_part_1, raw_timestamp_part_2, 
cvtd_timestamp, new_window, num_window) are time or identifying information 
that should be irrelevant to a prediction of the quality of exercise. Moreover, 
columns 12-36, 50-59, 69-83, 87-101, 103-112, 125-139, and 141-150 contain so 
many missing values that imputing the missing values would be unreasonable. No 
other columns have any missing values. Therefore, columns 1-7, 12-36,50-59, 
69-83, 87-101, 103-112, 125-139, and 141-150 were removed from both the testing 
and training files.


```r
## Remove all columns that are that are irrelevant to the outcome variable
## or contain a large number of missing values.

exercise_training <- exercise_training [-c(1:7,12:36,50:59,69:83,87:101,103:112,
                                           125:139,141:150)]
exercise_testing <- exercise_testing [-c(1:7,12:36,50:59,69:83,87:101,103:112,
                                         125:139,141:150)]
```

The model was fit using the random forest method in the caret package using all
default parameters. The random forest method is one of the most popular 
machine learning algorithms due to its high accuracy. The disadvantages are
the long computational time and the difficulty in interpreting the model
(as opposed to, for example, the tree classification method or the logistic
regression method). When the random forest method is used with the train
function, there is built-in cross-validation. The default cross-validation
method is bootstrap resampling with 25 replicates.


```r
## Fit the model using the random forest method using all default parameters.
## This includes cross-validation using bootstrap resampling with 25 replicates.

set.seed (1234)
modelFit <- train (exercise_training$classe ~ ., method = "rf", 
                   data = exercise_training)
print(modelFit)
```

```
## Random Forest 
## 
## 19622 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## Summary of sample sizes: 19622, 19622, 19622, 19622, 19622, 19622, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9927893  0.9908766
##   27    0.9927021  0.9907660
##   52    0.9843011  0.9801348
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 2.
```

```r
print (modelFit$finalModel)
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 2
## 
##         OOB estimate of  error rate: 0.41%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 5579    1    0    0    0 0.0001792115
## B   12 3782    3    0    0 0.0039504872
## C    0   21 3400    1    0 0.0064289889
## D    0    0   34 3180    2 0.0111940299
## E    0    0    0    6 3601 0.0016634322
```

The OOB (out-of-bag) estimate is an estimate of the out-of-sample error rate. 
In this case, the out-of-sample error rate is estimated as 0.44%.

The random forest model was then used to predict the classe variable for
the testing set.


```r
## Predict the classe values for the testing set.

prediction <- predict (modelFit, exercise_testing)
print (prediction)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

The predictions for the testing set had an accuracy of 100%. Since the
estimated out-of-sample error rate was so low and since the application to the
testing set was 100% successful, there was no need to consider any other
algorithm or any adjustment to the default parameters.


