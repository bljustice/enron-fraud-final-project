#Udacity Project 5 - Intro to Machine Learning Project
##Identify Fraud from Enron Email

1. This project is based on using supervised machine learning to predict if certain employees at Enron are POIs or not. The Enron dataset provided is a Python dictionary, which contains 146 top-level keys that are mostly employee names. The value for that key is another dictionary, which contains several different features about the employee, such as their salary, total stock value, number of emails to and from them, and many others. There are 18 employees classified as a POI, which means they were an Enron employee of interest for this particular dataset and 128 were classified as non-POIs. Also, many of the feature fields are incomplete in this dataset and include values such as ‘NaN’ or NoneType Python objects. I found 2 outliers while doing analysis on the dataset. One is ‘TOTAL’, which appears to be all of the datasets features combined together into one Python dictionary. The other is something that appears to be an organization called ‘THE TRAVEL AGENCY IN THE PARK’. I removed both of these from the dataset to avoid skewing my classification system’s results.

2. I used all of the features in the original dataset plus several of my own features in my final model. The only feature I didn't use was each person’s email address. When I started testing, I only added the following email-based features to my dataset.

  **Email Based Features**

  | Feature Name | Description |
  | ------------- | ------------- |
  | to_poi_percentage  | The percentage of emails the person sent to a POI  |
  | from_poi_percentage | The percentage of emails from a POI to a particular person |
  | total_poi_percentage | The percentage of total emails from a POI and to a POI from a particular person |

  I created the variables above because I thought that most POI employees would be communicating between one another more frequently than to non-POI employees. After running PCA without feature scaling, I found that 2 of the features explained approximately 21% of the variance in the dataset, so I decided to limit the number of features to 5 initially. Below is my model's recommended pipeline based on this setup, and f1 as the scoring metric. In section 3, I go into detail about why I used recall score instead after this initial test as the scoring metric for my algorithm.

  ```python
  Pipeline(steps=[('feat', PCA(copy=True, n_components=2, whiten=False)), ('classifier', GaussianNB())])
  ```
  Below are the results of this pipeline

  | Evaluation Metric | Score (rounded) |
  | ------------- | ------------- |
  | Accuracy | .87 |
  | Precision | .56 |
  | Recall | .28 |
  | F1 | .37 |

  Since this test did not perform to requirements, I decided to scale the dataset's features using MinMaxScaler, adding some additional new features I created, and switched my algorithm to use recall score as the deciding factor for choosing the most optimal pipeline. However before doing that, I tried adding the new features into my model with everything else staying constant to see how they affected performance. Below are the additional features I created and used.

  **New Email Based Feature**

  | Feature Name | Description |
  | ------------- | ------------- |
  | sqrt_total_poi_percentage | The square root value of the total_poi_percentage value |

  **Financial Based Features**

  | Feature Name | Description |
  | ------------- | ------------- |
  | total_employee_worth | The sum of a person’s salary, bonus, and total stock value |
  | log_total_employee_worth | The log based transformation of the total_employee_worth variable |

  I created the new email feature to transform the existing percentage metric and see if that affected success of my algorithm. I created the financial variables based on assumptions known about why Enron eventually went bankrupt and what I thought most likely made an employee a POI, which was that they would most likely have higher sums of financial value than non-POIs. Thus, I created the total_employee_worth variable. I also transformed it using a natural log to see how much it would help the overall success of my classification system. Turns out, these new features actually decreased the f1 score, but my model ended up picking a pipeline that had an increased recall score. Below are both the pipeline chosen and it's evaluation metrics.

  ```python
  Pipeline(steps=[('feat', PCA(copy=True, n_components=3, whiten=False)), ('classifier', AdaBoostClassifier(algorithm='SAMME.R',
          base_estimator=DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=None, splitter='best'),
          learning_rate=1.0, n_estimators=2, random_state=None))])
  ```

  | Evaluation Metric | Score (rounded) |
  | ------------- | ------------- |
  | Accuracy | .82 |
  | Precision | .32 |
  | Recall | .33 |
  | F1 | .33 |

  Based on this finding, I added in the rest of the model updates mentioned above including feature scaling and changing my scoring method to help create a more optimized model. I also updated the PCA step of my pipeline to possibly include a larger range of components to make sure I wasn't hindering overall performance. I used a random number of components between 1 and the total number of features being selected in my original feature list for final testing. Below are the model's scores using tester.py with scaled features, all the new features I created plus the original ones used, the larger component range, and scoring based on recall score. With these updates, it found K-Nearest Neighbors to be the best classification algorithm with two principal components.

  Below is the pipeline chosen by my algorithm with the new features and its results.

  ```python
  Pipeline(steps=[('scaler', MinMaxScaler(copy=True, feature_range=(0, 1))), ('feat', PCA(copy=True, n_components=2, whiten=False)), ('classifier', KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
             metric_params=None, n_jobs=1, n_neighbors=1, p=2,
             weights='uniform'))])
  ```

  | Evaluation Metric | Score (rounded) |
  | ------------- | ------------- |
  | Accuracy | .89 |
  | Precision | .61 |
  | Recall | .46 |
  | F1 | .52 |

  As seen above, precision score increased by 29%, recall score increased by 13%, and accuracy increased by 7% with these updates included. Based on the pipeline chosen by my algorithm, additional principal components didn't end up helping. Grid Search found that the best performing pipeline based on recall was with 2 scaled principal components.

3. My final classification model is built with a sklearn pipeline, starting with feature scaling, then PCA, then a classifier. Grid search cross validation was used to tune parameters for each step of the pipeline. Below are the classification algorithms I tried:

  - Random Forest
  - Decision Tree
  - Gaussian Naive Bayes
  - K-Nearest Neighbors
  - AdaBoost
  - Linear SVM

  When I first started testing, I didn't use feature scaling and I based the best algorithm off of the F1 score generated by the model. Precision score was quite high for each algorithm based on this setup, especially with Gaussian Naive Bayes and Random Forest classification, however recall was never above approximately 28%. The top performing pipeline based on these metrics was a Gaussian Naive Bayes pipeline, which can be seen in section 2 of this file. After adding in new features I made, the f1 score did not improve either. Because of this issue, I chose to scale features using MinMaxScaler and use recall as the scoring metric for the best algorithm. As a result, K-Nearest Neighbors ended up being the most optimal algorithm. Below is the final pipeline that was chosen:

  ```python
  Pipeline(steps=[('scaler', MinMaxScaler(copy=True, feature_range=(0, 1))), ('feat', PCA(copy=True, n_components=2, whiten=False)), ('classifier', KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
             metric_params=None, n_jobs=1, n_neighbors=1, p=2,
             weights='uniform'))])
  ```

4. Many machine learning algorithms have parameters that can be tuned to optimize their classification success. These parameters tend to be used to find a middle-ground between underfitting and overfitting a classification model to a dataset. If tuning of these parameters is not done well, a model can be very variated when trying to predict new data, or highly biased, meaning that it basically ignores new data features being used for prediction. This is called the bias-variance tradeoff. As stated above, I used grid search cross validation to exhaustively search a list of particular parameters for each pipeline I created.

5. Validation is a method of evaluating algorithms by splitting the dataset into training and testing sets. A classic mistake that can be made in validation is training and testing on the entire dataset. This can lead to algorithms showing very high accuracy results when predicting values on the dataset used to fit the model, but highly variated in prediction results of a new dataset. For my model, I chose to use stratified shuffle split cross validation. This type of validation splits data randomly into training and testing sets, each having distributions that reflect the overall dataset.

6. Below are the metrics of my classification model using tester.py.

  | Evaluation Metric | Score (rounded) |
  | ------------- | ------------- |
  | Accuracy | .89 |
  | Precision | .61 |
  | Recall | .46 |
  | F1 | .52 |

  The two evaluation metrics I was most concerned with are precision and recall. The precision score shows how many predicted POI employees are actually POIs. Alternatively, recall shows how many POI employees were predicted correctly out of the entire POI employee sample base. Based on the scores above, my model classified actual POI employees pretty well. However it also predicted several people that are actually non-POIs, as POIs. Below are the metrics of the confusion matrix outputted from tester.py.

  | Classified As Type | Number Classified |
  | ------------------ | ----------------- |
  | True Positives | 921 |
  | False Positives | 596 |
  | True Negatives | 12,404 |
  | False Negatives | 1,079 |   
