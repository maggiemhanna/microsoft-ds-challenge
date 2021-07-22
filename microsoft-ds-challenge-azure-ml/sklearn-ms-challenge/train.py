# -*- coding: utf-8 -*-
# Modified from https://www.geeksforgeeks.org/multiclass-classification-using-scikit-learn/

import argparse
import os

# importing necessary libraries
import pandas as pd
import numpy as np

from sklearn import metrics
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import joblib


from azureml.core.run import Run
from azureml.core import Dataset

run = Run.get_context()
ws = run.experiment.workspace

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_estimators', type=int, default=100,
                        help='The number of trees in the forest.')
    parser.add_argument('--criterion', type=str, default="gini",
                        help='The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain.')

    args = parser.parse_args()
    run.log('N estimators', np.int(args.n_estimators))
    run.log('Split criterion', np.str(args.criterion))

    # loading the datasets
    train_dataset = Dataset.get_by_name(ws, name='train')
    test_dataset = Dataset.get_by_name(ws, name='test')

    # load the TabularDataset to pandas DataFrame
    train = train_dataset.to_pandas_dataframe()
    test = test_dataset.to_pandas_dataframe()

    test = test[train.columns]

    # target column
    target_col = "EmployeeTargetedOverPastYear"

    # categorical ordinal features
    cat_ord_cols = ["Access Level", 
                    "behaviorPattern2", 
                    "peerUsageMetric6",
                    "usageMetric2", 
                    "usageMetric5",
                    "Social Media Activity (Scaled)"]

    # categorical nominal features
    cat_nom_cols = ['BD877Training Completed',
                    'Department Code', 
                    'Email Domain',
                    'fraudTraining Completed',
                    'Gender (code)']

    # no numerical features

    # X -> features, y -> label
    # X: explanatory variables / y: variable to predict
    X_train = train.drop(target_col, axis=1)
    y_train = train[[target_col]]

    X_test = test.drop(target_col, axis=1)
    y_test = test[[target_col]]

    # no numerical features
    # numeric_transformer = Pipeline(steps=[
    #     ('standardscaler', StandardScaler())])

    # ordinal features transformer
    ordinal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(missing_values=np.nan, strategy="most_frequent")),
        ('minmaxscaler', MinMaxScaler())
    ])

    # nominal features transformer
    # Bug: I couldn't use OneHotEncoder(drop="if_binary") as it doesn't work with explainer later
    nominal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(missing_values=np.nan, strategy="most_frequent")),
        ('onehot', OneHotEncoder(sparse=False))
    ])

    # imputer only for all other features
    imputer_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(missing_values=np.nan, strategy="most_frequent"))
    ])

    # preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('ordinal', ordinal_transformer, cat_ord_cols),
            ('nominal', nominal_transformer, ['Department Code', 'Email Domain']), # other features are already binary
            ('other', imputer_transformer, ['BD877Training Completed', 'fraudTraining Completed', 'Gender (code)'])], 
            remainder="passthrough")
                    
    # append classifier to preprocessing pipeline.
    # now we have a full prediction pipeline.
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', RandomForestClassifier(n_estimators =args.n_estimators, criterion = args.criterion, random_state=0))])


    # training a random forest classifier
    clf.fit(X_train, y_train)

    # predicting over training & testing datasets
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    # Model Training Accuracy, how often is the classifier correct?
    print("Train Accuracy: {:.2f}".format(metrics.accuracy_score(y_train, y_train_pred)))
    # Recall
    print("Train Recall: {:.2f}".format(metrics.recall_score(y_train, y_train_pred)))
    # Precision
    print("Train Precison: {:.2f}".format(metrics.precision_score(y_train, y_train_pred)))
    # F1score
    print("Train F1 Score: {:.2f}".format(metrics.f1_score(y_train, y_train_pred)))

    run.log('Train Accuracy', np.float(metrics.accuracy_score(y_train, y_train_pred)))
    run.log('Train Recall', np.float(metrics.recall_score(y_train, y_train_pred)))
    run.log('Train Precison', np.float(metrics.precision_score(y_train, y_train_pred)))
    run.log('Train F1 Score', np.float(metrics.f1_score(y_train, y_train_pred)))

    print("Train CM: ")
    print(metrics.confusion_matrix(y_train, y_train_pred))


    # Model Testing Accuracy, how often is the classifier correct?
    print("Test Accuracy: {:.2f}".format(metrics.accuracy_score(y_test, y_test_pred)))
    # Recall
    print("Test Recall: {:.2f}".format(metrics.recall_score(y_test, y_test_pred)))
    # Precision
    print("Test Precison: {:.2f}".format(metrics.precision_score(y_test, y_test_pred)))
    # F1score
    print("Test F1 Score: {:.2f}".format(metrics.f1_score(y_test, y_test_pred)))

    run.log('Test Accuracy', np.float(metrics.accuracy_score(y_test, y_test_pred)))
    run.log('Test Recall', np.float(metrics.recall_score(y_test, y_test_pred)))
    run.log('Test Precison', np.float(metrics.precision_score(y_test, y_test_pred)))
    run.log('Test F1 Score', np.float(metrics.f1_score(y_test, y_test_pred)))

    print("Test CM: ")
    print(metrics.confusion_matrix(y_test, y_test_pred))

    os.makedirs('outputs', exist_ok=True)
    # files saved in the "outputs" folder are automatically uploaded into run history
    joblib.dump(clf, 'outputs/model_rf.joblib') 


if __name__ == '__main__':
    main()
