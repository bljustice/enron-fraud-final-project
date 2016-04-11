#!/usr/bin/python
import sys
import pickle
from time import time
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
import math

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
### Task 2: Remove outliers
data_dict.pop('TOTAL',0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK',0)

### Task 3: Create new feature(s)
def create_new_features(data_dict):
    from_to_poi_ratios = []
    for k,v in data_dict.items():
        for key,value in v.items():
        #email features
            from_messages =  v['from_messages']
            from_poi = v['from_poi_to_this_person']
            to_poi = v['from_this_person_to_poi']
            to_messages = v['to_messages']
        #finanical features
            sals = v['salary']
            bonuses = v['bonus']
            total_stock = v['total_stock_value']
        #create new email features
            if from_messages != 'NaN' and from_poi != 'NaN' and to_poi != 'NaN':
                v['to_poi_percentage'] = float(to_poi)/float(from_messages)
                v['from_poi_percentage'] = float(from_poi)/float(to_messages)
                v['total_poi_percentage'] = (float(to_poi)+float(from_poi))/(float(from_messages)+float(to_messages))
                v['sqrt_total_poi_percentage'] = math.sqrt(v['total_poi_percentage'])
            else:
                v['to_poi_percentage'] = 'NaN'
                v['from_poi_percentage'] = 'NaN'
                v['total_poi_percentage'] = 'NaN'
                v['sqrt_total_poi_percentage'] = 'NaN'
        #create new finanical features
            if sals != 'NaN' and bonuses != 'NaN' and total_stock != 'NaN':
                 v['total_employee_worth'] = sals+bonuses+total_stock
                 v['log_total_employee_worth'] = math.log(v['total_employee_worth'])
            else:
                v['total_employee_worth'] = 'NaN'
                v['log_total_employee_worth'] = 'NaN'
    return data_dict

def define_features(data_dict):
    features_list = list(set(key for k,v in data_dict.items() \
    for key,value in v.items() if key != 'poi' and key != 'email_address'))
    features_list.insert(0,'poi')
    return features_list

### Store to my_dataset for easy export below.
my_dataset = data_dict
features_list = define_features(create_new_features(data_dict))

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
### Provided to give you a starting point. Try a variety of classifiers.
def build_classifier_list():
    clf_list = []
    rfc_clf = RandomForestClassifier()
    rfc_params = {'classifier__n_estimators':[1,2,3,5,15,20],'classifier__max_features':['auto','log2','sqrt']}
    clf_list.append((rfc_clf,rfc_params))
    dtree_clf = tree.DecisionTreeClassifier()
    dtree_params = {'classifier__min_samples_split':[2,3,5,15,20],'classifier__max_features':['auto','log2','sqrt']}
    clf_list.append((dtree_clf,dtree_params))
    gnb_clf = GaussianNB()
    gnb_params = {}
    clf_list.append((gnb_clf,gnb_params))
    knn_clf = KNeighborsClassifier()
    knn_params = {'classifier__n_neighbors':[1,2,3,5,15,20],'classifier__weights':['uniform','distance']}
    clf_list.append((knn_clf,knn_params))
    ada_clf = AdaBoostClassifier()
    ada_params = {'classifier__base_estimator':[tree.DecisionTreeClassifier()],'classifier__n_estimators':[1,2,3,5,15,20,50,100]}
    clf_list.append((ada_clf,ada_params))
    svm_clf = LinearSVC()
    svm_params = {'classifier__C':[.5,1,5,10,100,1000]}
    clf_list.append((svm_clf,svm_params))
    return clf_list

def build_scalers():
    all_scalers = []
    min_max = MinMaxScaler()
    min_max_params = {'scaler__feature_range':[(0,1)]}
    all_scalers.append((min_max,min_max_params))
    return all_scalers

def build_features():
    all_features = []
    pca = PCA()
    pca_params = {'feat__n_components':[1,2,3,5,15,20,len(features_list)-1]}
    all_features.append((pca,pca_params))
    return all_features

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
### Example starting point. Try investigating other evaluation techniques!
def build_grid_search(clf_list,feats,scalers):
    best_estimators_scores = []
    for clf in clf_list:
        for scaler in scalers:
            for feat in feats:
                params = {}
                params.update(scaler[1])
                params.update(feat[1])
                params.update(clf[1])
                pipe = Pipeline([('scaler',scaler[0]),('feat',feat[0]),('classifier',clf[0])])
                grid_search = GridSearchCV(pipe,param_grid=params,cv=StratifiedShuffleSplit(labels,50,random_state=42),scoring='recall')
                grid_search.fit(features,labels)
                best_estimators_scores.append((grid_search.best_estimator_,grid_search.best_score_))
    #Returns sorted list based on classifier recall score.
    return sorted(best_estimators_scores,key=lambda estimator: estimator[1],reverse=True)

clf = build_grid_search(build_classifier_list(),build_features(),build_scalers())[0][0]

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
dump_classifier_and_data(clf, my_dataset, features_list)
