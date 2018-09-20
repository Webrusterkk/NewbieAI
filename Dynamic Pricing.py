#!/usr/bin/env python
# coding: utf-8

# # Machine Learning 
# ## Supervised Learning
# ## Project: Dynamic Pricing Model

# Welcome to the  project of the Machine Learning ! In this notebook, some template code has already been provided for you, and it will be your job to implement the additional functionality necessary to successfully complete this project. Sections that begin with **'Implementation'** in the header indicate that the following block of code will require additional functionality which you must provide. Instructions will be provided for each section and the specifics of the implementation are marked in the code block with a `'TODO'` statement. Please be sure to read the instructions carefully!
# 
# In addition to implementing code, there will be questions that you must answer which relate to the project and your implementation. Each section where you will answer a question is preceded by a **'Question X'** header. Carefully read each question and provide thorough answers in the following text boxes that begin with **'Answer:'**. Your project submission will be evaluated based on your answers to each of the questions and the implementation you provide.  
# 
# >**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

# ### Question 1 - Classification vs. Regression
# *Your goal for this project is to identify students who might need early intervention before they fail to graduate. Which type of supervised learning problem is this, classification or regression? Why?*

# **Answer: **
# 
# Given that in this problem we aren't trying to predict continuous values, this is clearly not regression. This problem comes under classification. The problem statement gives a big clue as to why - we need to identify whether a student needs early intervention or not. So there are clearly two labels, wither of which could apply to a student - (a)whether a student needs early intervention, OR (b) A student doesn't need early intervention.
# 

# ## Exploring the Data
# Run the code cell below to load necessary Python libraries and load the student data. Note that the last column from this dataset, `'passed'`, will be our target label (whether the student graduated or didn't graduate). All other columns are features about each student.

# In[12]:


from time import time

import matplotlib.pyplot as plt
# Import libraries
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display  # Allows the use of display() for displaying DataFrames
from sklearn.metrics import *


pd.options.display.max_columns = None  # Allows us to view all columns of a DataFrame

# Read vessel data
vessel_data = pd.read_csv("MarineOcc_Eng.csv")

print(" data read successfully!")

# Display the first five records
#display(vessel_data.head(n=5))
# check whether columns is having any null or not
display(vessel_data.isnull().any())
# Some more additional data analysis
#display(np.round(vessel_data.describe()))

# ### Implementation: Data Exploration
# Let's begin by investigating the dataset to determine how many students we have information on, and learn about the graduation rate among these students. In the code cell below, you will need to compute the following:
# - The total number of students, `n_students`.
# - The total number of features for each student, `n_features`.
# - The number of those students who passed, `n_passed`.
# - The number of those students who failed, `n_failed`.
# - The graduation rate of the class, `grad_rate`, in percent (%).
# 

# In[13]:


# TODO: Calculate number of rows
n_rows = vessel_data.shape[0]

# TODO: Calculate number of columns
n_columns = vessel_data.shape[1]

print("The dataset has {} rows and {} columns".format(n_rows, n_columns))

#sns.countplot(y="OccClassID", data=vessel_data)
#plt.tight_layout()

# ## Preparing the Data
# In this section, we will prepare the data for modeling, training and testing.
# 
# ### Identify feature and target columns
# It is often the case that the data you obtain contains non-numeric features. This can be a problem, as most machine learning algorithms expect numeric data to perform computations with.
# 
# Run the code cell below to separate the student data into feature and target columns to see if any features are non-numeric.

# In[14]:


# plotting the dataset with a different color depending on the Casualities
df2 = vessel_data.loc[vessel_data["TotalFatalities"] == 2]
df3 = vessel_data.loc[vessel_data["TotalFatalities"] == 3]

xx2, yy2 = df2["Latitude"], -df2["Longitude"]
xx3, yy3 = df3["Latitude"], -df3["Longitude"]

#pts2 = plt.scatter(xx2, yy2, color='b')
#pts3 = plt.scatter(xx3, yy3, color='r')
#plt.legend((pts2, pts3), ('TotalFatalities= 2', 'TotalFatalities= 3'), loc='lower left')
#plt.title("Accident Severity Map")
#plt.tight_layout()

# In[15]:


# Extract feature columns
feature_cols = ['OccDate', 'OccYear', 'SeriousInjuries', 'TotalFatalities']

# Extract target column 'passed'
target_col = 'OccurrenceType'

# Show the list of columns
print("Feature columns:\n{}".format(feature_cols))
print("\nTarget column: {}".format(target_col))

# Separate the data into feature data and target data (X_all and y_all, respectively)
X_all = vessel_data[feature_cols]
y_all = vessel_data[target_col]

# Show the feature information by printing the first five rows
print("\nFeature values:")
display(X_all.head())


# ### Preprocess Feature Columns
# 
# As you can see, there are several non-numeric columns that need to be converted! Many of them are simply `yes`/`no`, e.g. `internet`. These can be reasonably converted into `1`/`0` (binary) values.
# 
# Other columns, like `Mjob` and `Fjob`, have more than two values, and are known as _categorical variables_. The recommended way to handle such a column is to create as many columns as possible values (e.g. `Fjob_teacher`, `Fjob_other`, `Fjob_services`, etc.), and assign a `1` to one of them and `0` to all others.
# 
# These generated columns are sometimes called _dummy variables_, and we will use the [`pandas.get_dummies()`](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html?highlight=get_dummies#pandas.get_dummies) function to perform this transformation. Run the code cell below to perform the preprocessing routine discussed in this section.

# In[16]:


def preprocess_features(X):
    """ Preprocesses the  data and converts N/A variables into
        Numberic 0 variables. Converts categorical variables into dummy variables. """

    # Initialize new output DataFrame
    output = pd.DataFrame(index=X.index)  # Empty DataFrame with range equal to X

    # Investigate each feature column for the data
    for col, col_data in X.iteritems():

        # print("col is ", col)

        # If data type is non-numeric, replace all yes/no values with 1/0

        if col == 'OccDate':
            col_data = col_data.apply(lambda x: int(x[5:7]))
            # print("col_data is ", col_data)
        if col == 'OccurrenceType':
            col_data = col_data.replace(['INCIDENT' , 'ACCIDENT'])
        # If data type is categorical, convert to dummy variables
        # if col_data.dtype == object:
        # Example: 'school' => 'school_GP' and 'school_MS'
        #   col_data = pd.get_dummies(col_data, prefix = col)

        # Collect the revised columns
        output = output.join(col_data)

    return output


X_all = preprocess_features(X_all)
print("Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns)))

# ### Implementation: Training and Testing Data Split
# So far, we have converted all _categorical_ features into numeric values. For the next step, we split the data (both features and corresponding labels) into training and test sets. In the following code cell below, you will need to implement the following:
# - Randomly shuffle and split the data (`X_all`, `y_all`) into training and testing subsets.
#   - Use 300 training points (approximately 75%) and 95 testing points (approximately 25%).
#   - Set a `random_state` for the function(s) you use, if provided.
#   - Store the results in `X_train`, `X_test`, `y_train`, and `y_test`.

# In[17]:


# TODO: Import any additional functionality you may need here

# Import train_test_split
from sklearn.model_selection import train_test_split

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
                                                    test_size=95,
                                                    random_state=0)

# Show the results of the split
print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))


# ## Training and Evaluating Models
# In this section, you will choose 3 supervised learning models that are appropriate for this problem and available in `scikit-learn`. You will first discuss the reasoning behind choosing these three models by considering what you know about the data and each model's strengths and weaknesses. You will then fit the model to varying sizes of training data (100 data points, 200 data points, and 300 data points) and measure the F<sub>1</sub> score. You will need to produce three tables (one for each model) that shows the training set size, training time, prediction time, F<sub>1</sub> score on the training set, and F<sub>1</sub> score on the testing set.
# 
# **The following supervised learning models are currently available in** [`scikit-learn`](http://scikit-learn.org/stable/supervised_learning.html) **that you may choose from:**
# - Gaussian Naive Bayes (GaussianNB)
# - Decision Trees
# - Ensemble Methods (Bagging, AdaBoost, Random Forest, Gradient Boosting)
# - K-Nearest Neighbors (KNeighbors)
# - Stochastic Gradient Descent (SGDC)
# - Support Vector Machines (SVM)
# - Logistic Regression

# ### Question 2 - Model Application
# *List three supervised learning models that are appropriate for this problem. For each model chosen*
# - Describe one real-world application in industry where the model can be applied. *(You may need to do a small bit of research for this — give references!)* 
# - What are the strengths of the model; when does it perform well? 
# - What are the weaknesses of the model; when does it perform poorly?
# - What makes this model a good candidate for the problem, given what you know about the data?

# **Answer: **
# 
# **Gaussian Naive Bayes**
# 
# * Can be used to mark emails as spam or not spam. Or classifying news articles
# * Performs well in classification problems, especially in supervised learning
# * The assumptions that Naive Bayes makes in conditional probabilities can often be inaccurate. It assumes that all features are independent of each other, while in the real world lot of these features can be dependent on each other
# * The data-set is not extremely large, and Gaussian Naive Bayes performs well under these conditions. Moreover, it's easy to train the classifier given that the set of features are not very large
# 
# **Decision Trees**
# 
# * Used in situations where the outputs are non binary. Can also be used in classification problems
# * Relatively easier to understand and interpret, since it is a 'white box' model
# * Prone to information gain, leading to high bias for some attributes
# * With the given data-set, there might be features that do not interact linearly. Decision trees can perform well in this regard. Moreover, combined with ensemble methods like bagging and boosting, they can be quite effective
# 
# **Random forest**
# 
# * Used for solving some of the overfitting problems in decision trees and other ML algorithms
# * Can handle non-binary classification problems quite well. Quite fast to train
# * With random sampling, all features might be treated equally and contribute to the final outcome
# * With this given data-set, random forest can be faster to train, can also handle large number of training examples
# 
# **Support Vector Machines**
# 
# * Used in text classification problems, face detection and in the health industry
# * Performs quite well in classification problems, and better than logistical regression methods. Guarantees to reach the global minimum rather than local minimum
# * Doesn't take into account the structure of the data or its order. Training time can get high with large datasets
# * It's less prone to overfitting some of the features in the data-set, and it can generalize well
# 
# Sources:
# 
# * [http://www.cs.cornell.edu/~caruana/ctp/ct.papers/caruana.icml06.pdf](http://www.cs.cornell.edu/~caruana/ctp/ct.papers/caruana.icml06.pdf)
# 
# * [https://en.wikipedia.org/wiki/Naive_Bayes_classifier](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)
# 
# * [https://en.wikipedia.org/wiki/Information_gain_in_decision_trees](https://en.wikipedia.org/wiki/Information_gain_in_decision_trees)
# 
# * [https://www.quora.com/What-are-the-advantages-of-different-classification-algorithms](https://www.quora.com/What-are-the-advantages-of-different-classification-algorithms)
# 

# ### Setup
# Run the code cell below to initialize three helper functions which you can use for training and testing the three supervised learning models you've chosen above. The functions are as follows:
# - `train_classifier` - takes as input a classifier and training data and fits the classifier to the data.
# - `predict_labels` - takes as input a fit classifier, features, and a target labeling and makes predictions using the F<sub>1</sub> score.
# - `train_predict` - takes as input a classifier, and the training and testing data, and performs `train_clasifier` and `predict_labels`.
#  - This function will report the F<sub>1</sub> score for both the training and testing data separately.

# In[26]:


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
    print("recall : %0.2f" % recall_score(y_test, y_pred,average='micro' ))
    print("precision : %0.2f" % precision_score(y_test, y_pred,average='micro'))
    print("f1 : %0.2f" % f1_score(y_test, y_pred,average='micro'))
    print("accuracy : %0.2f" % accuracy_score(y_test, y_pred))
    print("---------------------------------------------------")


# ### Implementation: Model Performance Metrics
# With the predefined functions above, you will now import the three supervised learning models of your choice and run the `train_predict` function for each one. Remember that you will need to train and predict on each classifier for three different training set sizes: 100, 200, and 300. Hence, you should expect to have 9 different outputs below — 3 for each model using the varying training set sizes. In the following code cell, you will need to implement the following:
# - Import the three supervised learning models you've discussed in the previous section.
# - Initialize the three models and store them in `clf_A`, `clf_B`, and `clf_C`.
#  - Use a `random_state` for each model you use, if provided.
#  - **Note:** Use the default settings for each model — you will tune one specific model in a later section.
# - Create the different training set sizes to be used to train each model.
#  - *Do not reshuffle and resplit the data! The new training points should be drawn from `X_train` and `y_train`.*
# - Fit each model with each training set size and make predictions on the test set (9 in total).  
# **Note:** Three tables are provided after the following code cell which can be used to store your results.

# In[24]:


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
