#!/usr/bin/env python
# coding: utf-8

# # Machine Learning
# ## Supervised Learning
# ## Project: Dynamic Pricing Model


from time import time
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display  # Allows the use of display() for displaying DataFrames
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import *
from sklearn.model_selection import train_test_split

pd.options.display.max_columns = None  # Allows us to view all columns of a DataFrame

# Read vessel data
vessel_data = pd.read_csv("47k.csv")

print(" data read successfully!")

display(vessel_data.info())

# TODO: Calculate number of rows
n_rows = vessel_data.shape[0]

# TODO: Calculate number of columns
n_columns = vessel_data.shape[1]

print("The dataset has {} rows and {} columns".format(n_rows, n_columns))


# Extract feature columns
feature_cols = ['YearBuilt', 'Gross Tonnage', 'Vessel Type','Vessel Sub Type', 'Hull Type', 'Weather','Wind Speed', 'Wind Direction','Sea State' ,'Departure' ,'Destination']

columns_to_format = ['Gross Tonnage', 'Vessel Type','Vessel Sub Type', 'Hull Type', 'Weather', 'Wind Direction','Sea State' ,'Departure' ,'Destination']

def format_data(col_name):
    global vectorizer
    vectorizer = TfidfVectorizer(min_df=5)
    vessel_data[col_name] = vessel_data[col_name].str.lower().replace('[^a-zA-Z0-9]', ' ', regex=True)
    vessel_data[col_name] = vectorizer.fit_transform(vessel_data[col_name])

for i in columns_to_format:
    format_data(i)
    print(i)


# Extract target column 'passed'
target_col = 'OccNo'

# Separate the data into feature data and target data (X_all and y_all, respectively)
X_all = vessel_data[feature_cols]
y_all = vessel_data[target_col]


def preprocess_features(X):
    """ Preprocesses the  data and converts N/A variables into
        Numberic 0 variables. Converts categorical variables into dummy variables. """

    # Initialize new output DataFrame
    output = pd.DataFrame(index=X.index)  # Empty DataFrame with range equal to X

    # Investigate each feature column for the data
    for col, col_data in X.iteritems():

        # print("col is ", col)

        # If data type is non-numeric, replace all yes/no values with 1/0

        # if col == 'OccDate':
        #     col_data = col_data.apply(lambda x: int(x[5:7]))
            # print("col_data is ", col_data)
        # if col == 'OccurrenceType':
        #     col_data = col_data.replace(['INCIDENT' , 'ACCIDENT'])
        # If data type is categorical, convert to dummy variables
        # if col_data.dtype == object:
        # Example: 'school' => 'school_GP' and 'school_MS'
        #   col_data = pd.get_dummies(col_data, prefix = col)

        # Collect the revised columns
        output = output.join(col_data)

    return output


X_all = preprocess_features(X_all)
print("Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns)))


# TODO: Import any additional functionality you may need here

# Import train_test_split


# TODO: Set the number of training points
num_train = 300

# Set the number of testing points
num_test = X_all.shape[0] - num_train

# TODO: Shuffle and split the dataset into the number of training and testing points above
X_train = None
X_test = None
y_train = None
y_test = None

# Split the 'features' and labels data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_all,
                                                    y_all,
                                                    test_size=9500,
                                                    random_state=0)

# Show the results of the split
print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))


def train_classifier(clf, X_train, y_train):
    ''' Fits a classifier to the training data. '''

    # Start the clock, train the classifier, then stop the clock

    start = time()
    clf.fit(X_train, y_train)
    end = time()

    # Print the results
    print("Trained model in {:.4f} seconds".format(end - start))


def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score. '''

    # Start the clock, make predictions, then stop the clock
    start = time()
    y_pred = clf.predict(features)
    end = time()

    # Print and return results
    print("Made predictions in {:.4f} seconds.".format(end - start))
    return printScores(target.values, y_pred, 'yes')
    #return printScores()


def train_predict(clf, X_train, y_train, X_test, y_test):
    ''' Train and predict using a classifer based on F1 score. '''

    # Indicate the classifier and the training set size
    print("Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train)))

    # Train the classifier
    train_classifier(clf, X_train, y_train)

    # Print the results of prediction for both training and testing



    print("F1 score for training set: {:.4f}.".format(predict_labels(clf, X_train, y_train)))
    print("F1 score for test set: {:.4f}.".format(predict_labels(clf, X_test, y_test)))


def printScores(y_test, y_pred, classif_name):
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    print("--------------  " + classif_name + "  ------------------")
    print("recall : %0.2f" % recall_score(y_test, y_pred ))
    print("precision : %0.2f" % precision_score(y_test, y_pred))
    print("f1 : %0.2f" % f1_score(y_test, y_pred))
    print("accuracy : %0.2f" % accuracy_score(y_test, y_pred))
    print("---------------------------------------------------")




# TODO: Import the three supervised learning models from sklearn
# from sklearn import model_A
# from sklearn import model_B
# from sklearn import model_C
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# TODO: Initialize the three models
#clf_A = GaussianNB()
#clf_B = DecisionTreeClassifier(max_depth=None, random_state=None)
clf_C = RandomForestClassifier(max_depth=None, random_state=None)

# le.fit(clf_C)


# TODO: Set up the training set sizes
X_train_100 = X_train[:100]
y_train_100 = y_train[:100]

X_train_200 = X_train[:200]
y_train_200 = y_train[:200]

X_train_300 = X_train[:300]
y_train_300 = y_train[:300]

X_samples = [X_train_100, X_train_200, X_train_300]
y_samples = [y_train_100, y_train_200, y_train_300]
# TODO: Execute the 'train_predict' function for each classifier and each training set size
for clf in [clf_C]:
    clf_name = clf.__class__.__name__

    for i, samples in enumerate(X_samples):
        train_predict(clf, samples, y_samples[i], X_test, y_test)

# train_predict(clf, X_train, y_train, X_test, y_test)


# ### Question 3 - Final F<sub>1</sub> Score
# *What is the final model's F<sub>1</sub> score for training and testing? How does that score compare to the untuned model?*

# **Answer: **
#
# As it turns out, the final model's performance has infact slightly increased, though not by a huge margin. The final model's F1 score is 80.5%.

# > **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to
# **File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.
