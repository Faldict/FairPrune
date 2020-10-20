
from utils import *
from datetime import datetime
import numpy as np
import pandas as pd
from fairlearn.reductions import GridSearch
from fairlearn.reductions import DemographicParity, EqualizedOdds
from aif360.sklearn.preprocessing import Reweighing

__all__ = ['pruning', 'unconstraint', 'reweighing', 'lagrangian']

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
models = {
    'LR': LogisticRegression(solver='liblinear', fit_intercept=True),
    'SVM': SVC(gamma='auto'),
    'MLP': MLPClassifier(
        hidden_layer_sizes=(100,), random_state=1, max_iter=300)
}

def time_diff_in_microseconds(time_diff):
    """ Convert time difference into micro seconds
    """
    return time_diff.microseconds + (10 ** 6) * time_diff.seconds

def pruning(num_dropped_features, constraint, model, lambda_value, X_train, X_test, Y_train, A_train):
    """ Conduct pruning algorithm and then use base classifier to train and predict.
    """
    start_time = datetime.now()
    sorted_by_mutual_info = compute_mutual_information_between_X_and_A(X_train, Y_train, A_train, constraint, lambda_value)
    clf = models[model]
    clf.fit(X_train[sorted_by_mutual_info[:len(X_train.columns) - num_dropped_features]], Y_train)
    Y_pred = clf.predict(X_test[sorted_by_mutual_info[:len(X_train.columns) - num_dropped_features]])
    end_time = datetime.now()
    return Y_pred, time_diff_in_microseconds(end_time-start_time)


def unconstraint(model, X_train, X_test, A_train, A_test, Y_train):
    """ Use the base classifier to train and predict without adding any constraints.
    """
    start_time = datetime.now()
    clf = models[model]
    clf.fit(np.array(combine_X_A_as_dataframe(X_train, A_train)), Y_train)
    Y_pred = clf.predict(np.array(combine_X_A_as_dataframe(X_test, A_test)))
    end_time = datetime.now()
    return Y_pred, time_diff_in_microseconds(end_time-start_time)


def reweighing(model, X_train, A_train, Y_train, X_test):
    """ Conduct reweighing algorithm and then use base classifier to train and predict.
    """
    start_time = datetime.now()
    rew = Reweighing('sensitive_attribute')
    X_A_train = combine_X_A_as_dataframe(X_train, A_train)
    _, sample_weight = rew.fit_transform(X_A_train, Y_train)
    end_time = datetime.now()
    clf = models[model]
    clf.fit(X_train, Y_train, sample_weight=sample_weight)
    Y_pred = clf.predict(X_test)
    
    return Y_pred, time_diff_in_microseconds(end_time-start_time)


def lagrangian(constraint, model, constraint_weight, grid_size, X_train, Y_train, A_train, X_test):
    """ Conduct lagrangian algorithm and set the base classifier as the black-box 
    estimator to train and predict.
    """
    start_time = datetime.now()
    if constraint == 'DP':
            clf = GridSearch(
                models[model],
                constraints=DemographicParity(),
                constraint_weight=constraint_weight,
                grid_size=grid_size)
    elif constraint == 'EO':
        clf = GridSearch(
            models[model],
            constraints=EqualizedOdds(),
            constraint_weight=constraint_weight,
            grid_size=grid_size)
    clf.fit(X_train, Y_train, sensitive_features=A_train)
    Y_pred = clf.predict(X_test)
    end_time = datetime.now()
    return Y_pred, time_diff_in_microseconds(end_time-start_time)

