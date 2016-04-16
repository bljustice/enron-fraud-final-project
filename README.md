#Udacity Project 5 - Intro to Machine Learning Project
##Identify Fraud from Enron Email

1. This project is based on using supervised machine learning to predict if certain employees at Enron are POIs or not. The Enron dataset provided is a Python dictionary, which contains 146 top-level keys that are mostly employee names. The value for that key is another dictionary, which contains several different features about the employee, such as their salary, total stock value, number of emails to and from them, and many others. There are 18 employees classified as a POI, which means they were an Enron employee of interest for this particular dataset and 128 were classified as non-POIs. Also, many of the feature fields are incomplete in this dataset and include values such as ‘NaN’ or NoneType Python objects. I found 2 outliers while doing analysis on the dataset. One is ‘TOTAL’, which appears to be all of the datasets features combined together into one Python dictionary. The other is something that appears to be an organization called ‘THE TRAVEL AGENCY IN THE PARK’. I removed both of these from the dataset to avoid skewing my classification system’s results.

2. From a high level, my final model creates multiple sklearn pipelines including the following steps:

  1. Feature scaling using MinMaxScaler
  2. PCA on scaled features
  3. Classification algorithm

  Grid search cross validation is used to tune the parameters of each step of each pipeline created. Once each parameter combination has been fitted and tested, my model returns the highest scoring pipeline based on the score type set in Grid Search CV. Originally I used F1 score as the score type, but switched to recall in later testing phases. The reason for changing this is explained later in section 3 of this report. Due to this type of setup, different pipelines can be returned when updates are made to the model, such as features being added or removed, and tuning parameters being adjusted. This is why different pipelines are shown below when new features are added to the model.

  When starting this project, I used PCA without feature scaling. I found that 2 of the features explained approximately 21% of the variance in the dataset, so I decided to limit the number of features to 5 in the first few tests and use Grid Search CV to run through each combination of parameters. Below is the initial testing pipeline chosen by my model based on this and only using the features provided in the original dataset minus each person's email address.

  ```python
  Pipeline(steps=[('feat', PCA(copy=True, n_components=2, whiten=False)), ('classifier', GaussianNB())])
  ```

  | Evaluation Metric | Score (rounded) |
  | ------------- | ------------- |
  | Accuracy | .87 |
  | Precision | .56 |
  | Recall | .28 |
  | F1 | .37 |


  Since this did not perform up to requirements, I added the following self-engineered email-based features to the dataset.

  **Email Based Features**

  | Feature Name | Description |
  | ------------- | ------------- |
  | to_poi_percentage  | The percentage of emails the person sent to a POI  |
  | from_poi_percentage | The percentage of emails from a POI to a particular person |
  | total_poi_percentage | The percentage of total emails from a POI and to a POI from a particular person |

  I created the variables above because I thought that most POI employees would be communicating between one another more frequently than to non-POI employees. Below is my model's chosen pipeline based on this setup, and F1 as the scoring metric.

  ```python
  Pipeline(steps=[('feat', PCA(copy=True, n_components=2, whiten=False)), ('classifier', GaussianNB())])
  ```

  | Evaluation Metric | Score (rounded) |
  | ------------- | ------------- |
  | Accuracy | .87 |
  | Precision | .56 |
  | Recall | .28 |
  | F1 | .37 |

  Since this test did not perform to requirements and the features appeared to have no effect on the evaluation metrics I was using, I decided to create additional features to use in my model with everything else staying constant to see how they affected performance. Below are the additional features I created and used.

  **New Email Based Feature**

  | Feature Name | Description |
  | ------------- | ------------- |
  | sqrt_total_poi_percentage | The square root value of the total_poi_percentage value |

  **Financial Based Features**

  | Feature Name | Description |
  | ------------- | ------------- |
  | total_employee_worth | The sum of a person’s salary, bonus, and total stock value |
  | log_total_employee_worth | The log based transformation of the total_employee_worth variable |

  I created the new email feature to transform the existing percentage metric and see if that affected success of my algorithm. I created the financial variables based on assumptions known about why Enron eventually went bankrupt and what I thought most likely made an employee a POI, which was that they would most likely have higher sums of financial value than non-POIs. Thus, I created the total_employee_worth variable. I also transformed it using a natural log to see how much it would help the overall success of my classification system. Below is the pipeline chosen by my model and its evaluation metrics based on this.

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

  As seen above, my model selected a different pipeline based on adding these new features. This new pipeline actually lowered the F1 and precision scores, but increased the recall score slightly. This pipeline met specifications, however I wanted to see if I could make an even better model. I decided to scale the dataset's features using MinMaxScaler, add all of the self-engineered features above, and switch my algorithm to use recall score as the deciding factor for choosing the most optimal pipeline. I also updated the PCA step of my pipeline to possibly include a larger range of components to make sure I wasn't hindering overall performance. I used a random number of components between 1 and the total number of features being selected in my original feature list, with Grid Search CV to test each number of principal components in each pipeline created. With these updates, it found K-Nearest Neighbors to be the best classification algorithm with two principal components. The final pipeline chosen by my model based on these updates has been provided in section 3 of this report and its evaluation metrics in section 6.

3. As stated in section 2, my final classification model builds multiple sklearn pipelines, starting with feature scaling, then PCA, then a classifier. Once a pipeline is built, it's parameters are tuned using Grid Search CV to find optimal performance based on the scoring method selected. Below are the classification algorithms I tried:

  - Random Forest
  - Decision Tree
  - Gaussian Naive Bayes
  - K-Nearest Neighbors
  - AdaBoost
  - Linear SVM

  When I first started testing, I didn't use feature scaling and I based the best algorithm off of the F1 score generated by the model. The top performing pipeline based on this scoring method was a Gaussian Naive Bayes pipeline, which can be seen in section 2 of this report. After adding in new features I made, the pipeline selected by my model did not improve the F1 score, but it met specifications. However, I wanted to continue to increase the effectiveness of my model. Because of this, I chose to add feature scaling, a different range of possible principal components with Grid Search CV for parameter tuning, and use recall as the scoring metric for the best pipeline. I noticed that recall score during the first few tests was rather low, and based on the way I constructed my model, I thought that a recall-based score may be a better fit for pipeline selection. As a result, a pipeline using K-Nearest Neighbors was chosen as the most optimal pipeline. Below is the final pipeline that was chosen by my model:

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
