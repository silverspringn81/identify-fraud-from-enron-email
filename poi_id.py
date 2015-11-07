#!/usr/bin/python
######################################################################################################
# This program will use machine learning techniques to build an algorithm to identify Enron Employees
# who may have committed fraud (POIs) based on the public Enron financial and email dataset.
# It consists of the following steps:
# - explore the background of the dataset
# - select and create features using Decision Tree
# - investigate and remove outliers using Gaussian NB
# - feature scaling (and PCA)
# - initial try of three machine learning algorithms
# - grid search to tune the parameters for three algorithms
# - show the best parameters
# - create pickle files
#
# Some of the processes are commented out in the default code and need to be uncommented when used:
# - outlier investigation
# - feature importance used in feature selection
# - initial try of three machine learning algorithms
# - tuning process with PCA
# - two out of the three algorithms that are not used
#
# Updated on Nov. 7, 2015
# 
######################################################################################################
import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
import matplotlib.pyplot
from tester import test_classifier, dump_classifier_and_data


# Define financial, email features & poi features
FINANCIAL_FEATURES = ['salary', 'deferral_payments', 'total_payments', 'loan_advances',\
                      'bonus','restricted_stock_deferred','deferred_income', 'total_stock_value',\
                      'expenses','exercised_stock_options', 'other', 'long_term_incentive',\
                      'restricted_stock', 'director_fees'] 
EMAIL_FEATURES = ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages',\
                  'from_this_person_to_poi', 'shared_receipt_with_poi'] 

POI_LABEL = ['poi']

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

# Feature selection consists of the following steps
'''
1. Input all the 14 financial features and run Decision Tree algorithm
2. Select most important features using feature_importances_ attribute of Decision Tree classifier
3. Figure out performance change of algorithm using different number of important features
4. Decide which of the financial features should be used.
5. Engineer new fractional features from original email features
6. Include the new features one by one and see if the performance improves
7. If the performance improves, add that feature to the list
'''

# features list starts from `poi'
features_list = POI_LABEL
#features_list.extend(FINANCIAL_FEATURES)

# Top-5 important financial features
top_features = ['exercised_stock_options', 'total_payments', 'bonus', 'restricted_stock', 'long_term_incentive']

# New features created from email features
new_features = ['fraction_from_poi','fraction_to_poi','fraction_shared_with_poi']

# The best features found by feature selection.
# They are used in the tuning process.
features_list.extend(top_features[0:3])
features_list.extend(new_features[2:3])

# Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

### Data exploration 
print "----Data Exploration----"
print "Number of person-feature pairs:", len(data_dict)
print "First 5 persons:", data_dict.keys()[:5]
print "Number of features:", len(data_dict[data_dict.keys()[0]])
print "First 5 features:", data_dict[data_dict.keys()[0]].keys()[:5]


# Count number and create list of POIs 
n_poi = 0
poi_list = []
for key in data_dict:
    if data_dict[key]["poi"] == True:
        n_poi += 1
        poi_list.append(key)

print "Number of POIs:", n_poi
print "List of POIs:", poi_list
print " "


# Find features with many missing or NaN values
print "----Check Feature Values----"
miss_values = ["","NaN"," "]
f_list = data_dict[data_dict.keys()[0]].keys()
miss_cnt_list = []

for f in f_list:
    miss_cnt = 0
    for key in data_dict:
        if data_dict[key][f] in miss_values:
            miss_cnt += 1
        
    miss_cnt_list.append((f, miss_cnt))

print "Features with more than 1/2 missing values..."
print [t for t in miss_cnt_list if t[1] > len(data_dict)/2]
print " "


### Task 2: Remove outliers
fin_features_pair = []

# Create many 2D plots with financial features to detect outliers

# These lines are to see effect of removing outliers on the plots
'''
data_dict.pop("TOTAL", 0)
data_dict.pop("BHATNAGAR SANJAY",0)
'''

'''
print "---Plot Financial Data to Detect Outliers---"
print " "
for i in range(len(FINANCIAL_FEATURES)-1):
    fin_features_pair = [FINANCIAL_FEATURES[i], FINANCIAL_FEATURES[i+1]]
    data = featureFormat(data_dict, fin_features_pair)
    
    for point in data:
        x = point[0]
        y = point[1]
        matplotlib.pyplot.scatter(x, y)
        
    matplotlib.pyplot.xlabel(FINANCIAL_FEATURES[i])
    matplotlib.pyplot.ylabel(FINANCIAL_FEATURES[i+1])

    ### Save plots to eps files
    ### for 'TOTAL'
    if i == 0: 
        matplotlib.pyplot.savefig('fig/outlier_TOTAL.png')
    
    ### for 'BHATNAGAR SANJAY'
    #if i == 4: 
    #    matplotlib.pyplot.savefig('fig/outlier_SANJAY.eps')
    
    ### for 'BHATNAGAR SANJAY'
    #if i == 2:
    #    matplotlib.pyplot.savefig('fig/outlier_KENNETH.eps')
    
    matplotlib.pyplot.show()
'''
    
    
# Investigate outlier candidates
print "---Investigate Outlier Candidate---"
print "TOTAL:", data_dict["TOTAL"]
print " "
print "BHATNAGAR SANJAY:", data_dict["BHATNAGAR SANJAY"]
print " " 
print "LAY KENNETH L:", data_dict["LAY KENNETH L"]
print " "
print "THE TRAVEL AGENCY IN THE PARK:", data_dict["THE TRAVEL AGENCY IN THE PARK"]
print " "


# Remove detected Outliers
print "---Removing Outliers---"
data_dict.pop("TOTAL", 0)
data_dict.pop("BHATNAGAR SANJAY",0)
data_dict.pop("THE TRAVEL AGENCY IN THE PARK",0)
print " "


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

# Supplement method to compute fraction from email features
def computeFraction(poi_messages, all_messages):
    fraction = 0
    if not "NaN" in [poi_messages, all_messages]:
        fraction = float(poi_messages) / all_messages
    else:
        fraction = 0

    return fraction


my_dataset = data_dict

# Add three new features created from email features to dataset
print "---Creating New Features---"
print " "
print "---Checking Outliers in New Features--- "

for name in my_dataset:
    data_point = my_dataset[name]

    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    shared_receipt_with_poi = data_point["shared_receipt_with_poi"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )
    fraction_shared_with_poi = computeFraction( shared_receipt_with_poi, to_messages )
    data_point["fraction_from_poi"] = fraction_from_poi
    data_point["fraction_shared_with_poi"] = fraction_shared_with_poi
    
    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = computeFraction(from_this_person_to_poi, from_messages )
    data_point["fraction_to_poi"] = fraction_to_poi
    
    my_dataset[name]["fraction_from_poi"] = fraction_from_poi
    my_dataset[name]["fraction_to_poi"] = fraction_to_poi
    my_dataset[name]["fraction_shared_with_poi"] = fraction_shared_with_poi

    
print " "
    
# Explore outliers in the newly-created feature by using 2D plots
'''
print "---Plot Newly-Created Features to Detect Outliers---"
print " "
data = featureFormat(my_dataset, ["fraction_from_poi", "fraction_to_poi"])
for point in data:
    x = point[0]
    y = point[1]
    matplotlib.pyplot.scatter(x, y)

matplotlib.pyplot.xlabel("from_poi")
matplotlib.pyplot.ylabel("to_poi")
matplotlib.pyplot.show()

data = featureFormat(my_dataset, ["fraction_from_poi", "fraction_shared_with_poi"])
for point in data:
    x = point[0]
    y = point[1]
    matplotlib.pyplot.scatter(x, y)

matplotlib.pyplot.xlabel("from_poi")
matplotlib.pyplot.ylabel("fraction_shared_with_poi")
matplotlib.pyplot.show()    
'''



### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

print "---Features Used in the Following Learning---"
print features_list
print " "

### Task 4: Try a variety of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Import pipeline for PCA and feature scaling
from sklearn.pipeline import Pipeline
# Import GridSearch
from sklearn.grid_search import GridSearchCV
# Import PCA
from sklearn.decomposition import RandomizedPCA
# Import MinMaxScaler for feature scaling
from sklearn.preprocessing import MinMaxScaler

### I use Gaussian Naive Bayes, Decision Tree, and Support Vector Machine.
### Feature scaling and PCA are carried out before every algorithm for the first tuning process.
### Several parameters are explored for PCA, Decision Tree and SVM.
### PCA process is eliminated for finer tuning. 

# Gaussian NB
from sklearn.naive_bayes import GaussianNB

# This line is for outlier removal and feature selection process
#clf = GaussianNB() 

# Tuning with PCA
'''
clf = Pipeline([('scaler' , MinMaxScaler()), ('reduce_dim', RandomizedPCA()), ('gnb', GaussianNB())])
parameters = {'reduce_dim__n_components': (2,3,4)} 
'''

# Tuning without PCA
'''
clf = Pipeline([('scaler' , MinMaxScaler()), ('gnb', GaussianNB())])
parameters = {} 
'''

# Decision Tree
from sklearn.tree import DecisionTreeClassifier

# This line is for feature selection
#clf = DecisionTreeClassifier()

# Tuning with PCA
'''
clf = Pipeline([('scaler' , MinMaxScaler()), ('reduce_dim', RandomizedPCA()), ('dct', DecisionTreeClassifier())])
parameters = {'reduce_dim__n_components': (2,3,4), 'dct__min_samples_split': (2,3,5)}
'''

# Tuning without PCA
'''
clf = Pipeline([('scaler' , MinMaxScaler()), ('dct', DecisionTreeClassifier())])
parameters = {'dct__min_samples_split': (1,2,3,4)}
'''

# SVM
from sklearn.svm import SVC

# Tuning with PCA
'''
clf = Pipeline([('scaler' , MinMaxScaler()), ('reduce_dim', RandomizedPCA()), ('svm', SVC())])
parameters = {'reduce_dim__n_components': (2,3,4), 
              'svm__C': (10, 100, 1000),
              'svm__gamma': (10,100, 1000)}
'''

# Tuning without PCA

clf = Pipeline([('scaler' , MinMaxScaler()), ('svm', SVC())])
parameters = {'svm__C': (1000, 10000, 100000),
              'svm__gamma': (1, 5, 10)}


# The followings are for initial try of each algorithm without gird search
'''
#pipeline = Pipeline([('scaler' , MinMaxScaler()), ('reduce_dim', RandomizedPCA()), ('gnb', GaussianNB())])
#pipeline = Pipeline([('scaler' , MinMaxScaler()), ('reduce_dim', RandomizedPCA()), ('dct', DecisionTreeClassifier(min_samples_split=2))])
pipeline = Pipeline([('scaler' , MinMaxScaler()), ('reduce_dim', RandomizedPCA()), ('svm', SVC(C=10, gamma=10))])
clf = pipeline
'''



### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


print "---Train/Test Data and Search Best Algorithm-Tune Combination---"
print " "

# GridSearch and tune parameters with F1 score
# The lines below should be commented out when exploring outliers or selecting features

from sklearn import cross_validation

cv = cross_validation.StratifiedShuffleSplit(labels, 3, random_state=42)
a_grid_search = GridSearchCV(clf, param_grid = parameters, cv=cv, scoring='f1')
a_grid_search.fit(features, labels)
best_clf = a_grid_search.best_estimator_

# Get best parameters after GridSearch
print " "
print "---Best Parameters Found---"
print  a_grid_search.best_params_
print " " 

test_classifier(best_clf, my_dataset, features_list)
print " "


# This line is used to figure out feature importances in feature selection process
'''
print "---Feature Importances---"
fi_array = []
for i in range(len(FINANCIAL_FEATURES)):
    print FINANCIAL_FEATURES[i], clf.feature_importances_ [i]
'''




### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)


