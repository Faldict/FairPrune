# This experiment will compare reduction approach and our
# approach on Adult dataset with noise.

import numpy as np
import pandas as pd
import json
import responsibly
from sklearn.model_selection import train_test_split
from datasets import *
import argparse
from fairlearn.reductions import GridSearch, ErrorRate
from fairlearn.metrics import demographic_parity_difference
from fairlearn.reductions import GridSearch, ErrorRate
from utils import mutual_information_2d, equal_opportunity_difference
import matplotlib.pyplot as plt
import seaborn as sns
from fairlearn.reductions import DemographicParity, TruePositiveRateParity
sns.set()
plt.style.use('seaborn')

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
models = {
    'LR': LogisticRegression(solver='liblinear', fit_intercept=True),
    'SVM': SVC(gamma='auto'),
    'MLP': MLPClassifier(
        hidden_layer_sizes=(100,), random_state=1, max_iter=300)
}


def import_dataset(dataset_name):
    print('importing dataset ' + dataset_name)
    if dataset_name == 'compas':
        X, Y, A = compas()
    else:
        X, Y, A = adult()
    return X, Y, A
    # return X[:100] + Y[:100] + A[:100]


def add_noise(noise_rate, sensitive_features):
    print('adding noise to sensitive features with noise_rate = ' +
          str(noise_rate))
    labels = np.unique(sensitive_features)
    assert len(labels) == 2
    m = len(sensitive_features)
    random_number_array = np.random.rand(m)
    for i in range(m):
        if noise_rate > random_number_array[i]:
            if sensitive_features[i] == labels[0]:
                sensitive_features[i] = labels[1]
            elif sensitive_features[i] == labels[1]:
                sensitive_features[i] = labels[0]
    return sensitive_features


def compute_mutual_information_between_X_and_A(X, Y, A, args):
    mis = []
    # TODO(yzhu): delete it
    if args.use_all_XYA:
        X, Y, A = import_dataset(args.dataset)
    for col in X.columns:
        if args.constraint == 'EO':
            mi = mutual_information_2d(X[col].values[Y == 0],
                                       A[Y == 0]) + mutual_information_2d(
                                           X[col].values[Y == 1], A[Y == 1])
        else:
            mi = mutual_information_2d(
                X[col].values, A) - args.lambda_value * mutual_information_2d(
                    X[col].values, Y)
        mis.append((mi, col))
    mis = sorted(mis, reverse=False)
    mis1 = [l[1] for l in mis]
    return mis1


def run_pipeline(args):
    X, Y, A = import_dataset(args.dataset)
    reports = {
        "reduction": {
            "accuracy": {},
            "dp": {},
            "eo": {}
        },
        "ours": {
            "accuracy": {},
            "dp": {},
            "eo": {}
        }
    }
    for i in range(args.repeats):
        print('Repeat #' + str(i))
        X_train, X_test, Y_train, Y_test, A_train, A_test = train_test_split(
            X,
            Y,
            A,
            test_size=0.2,
            random_state=131 + i,
            stratify=np.stack((Y, A), axis=1))
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        A_train = add_noise(args.noise_rate, A_train)
        print('Running our experiment ... ')
        run_ours_experiment(args, reports, X_train, X_test, Y_train, Y_test,
                            A_train, A_test)
        print('Running reduction experiment ... ')
        run_reduction_experiment(args, reports, X_train, X_test, Y_train,
                                 Y_test, A_train, A_test)

    print('post processing ... ')
    print(reports)
    post_processing(reports, args)

    print('output results ... ')
    with open('results/feature_noise_compare_' + str(args) + '.json',
              'w') as fp:
        json.dump(reports, fp)


def post_processing(reports, args):
    if args.use_all_XYA:
        with open(
                'results/user_all_XYA.json',
                'r') as fp:
            to_be_saved = json.load(fp)
    else:
        with open(
                'results/user_train_XYA.json',
                'r') as fp:
            to_be_saved = json.load(fp)

    
    for experiment in reports:
        for matrix in reports[experiment]:
            res = []
            for k in reports[experiment][matrix]:
                res.append(np.array(reports[experiment][matrix][k]).mean())
            reports[experiment][matrix] = res


def run_reduction_experiment(args, reports, X_train, X_test, Y_train, Y_test,
                             A_train, A_test):
    constraint_weights = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    # constraint_weights = [0.0, 1.0]
    for constraint_weight in constraint_weights:
        print('Running for constraint weight = ' + str(constraint_weight))
        if args.constraint == 'DP':
            clf = GridSearch(
                models[args.model],
                constraints=DemographicParity(),
                constraint_weight=constraint_weight,
                grid_size=args.grid_size)
        elif args.constraint == 'EO':
            clf = GridSearch(
                models[args.model],
                constraints=TruePositiveRateParity(),
                constraint_weight=constraint_weight,
                grid_size=args.grid_size)
        print('Fitting ...')
        clf.fit(X_train, Y_train, sensitive_features=A_train)
        print('Predicting ...')
        Y_pred = clf.predict(X_test)

        print('Calculating accuracy, DP, and EO ...')
        if constraint_weight not in reports["reduction"]["accuracy"]:
            reports["reduction"]["accuracy"][constraint_weight] = []
            reports["reduction"]["dp"][constraint_weight] = []
            reports["reduction"]["eo"][constraint_weight] = []
        reports['reduction']['accuracy'][constraint_weight].append(
            (Y_pred == Y_test).mean())
        reports['reduction']['dp'][constraint_weight].append(
            demographic_parity_difference(
                y_true=Y_test, y_pred=Y_pred, sensitive_features=A_test))
        reports['reduction']['eo'][constraint_weight].append(
            equal_opportunity_difference(
                Y_true=Y_test, Y_pred=Y_pred, sensitive_features=A_test))


def run_ours_experiment(args, reports, X_train, X_test, Y_train, Y_test,
                        A_train, A_test):
    sorted_by_mutual_info = compute_mutual_information_between_X_and_A(
        X_train, Y_train, A_train, args)
    for i in range(args.n):
        clf = models[args.model]
        clf.fit(X_train[sorted_by_mutual_info[:len(X_train.columns) - i]],
                Y_train)
        Y_pred = clf.predict(
            X_test[sorted_by_mutual_info[:len(X_train.columns) - i]])
        if i not in reports["ours"]["accuracy"]:
            reports["ours"]["accuracy"][i] = []
            reports["ours"]["dp"][i] = []
            reports["ours"]["eo"][i] = []
        reports["ours"]["accuracy"][i].append((Y_pred == Y_test).mean())
        reports["ours"]["dp"][i].append(
            demographic_parity_difference(
                y_true=Y_test, y_pred=Y_pred, sensitive_features=A_test))
        reports["ours"]["eo"][i].append(
            equal_opportunity_difference(
                Y_true=Y_test, Y_pred=Y_pred, sensitive_features=A_test))


def clean_graph(disparity, accuracy):
    pareto = [(accuracy[i], disparity[i]) for i in range(len(disparity))]
    pareto = sorted(pareto)

    x = [pareto[0][0]]
    y = [pareto[0][1]]
    for i in range(1, len(disparity)):
        if pareto[i][1] >= y[-1]:
            x.append(pareto[i][0])
            y.append(pareto[i][1])
    return y, x
    # return disparity, accuracy


def plot_result():
    print('printing ' + str(args) + '...')
    plt.clf()
    with open(
            'results/feature_noise_compare_' + str(args) + '.json',
            'r') as fp:
        reports = json.load(fp)

    x, y = clean_graph(reports['ours']['eo'], reports['ours']['accuracy'])
    plt.plot(
        x, 1. - np.array(y), '^-', color='red', label='Ours, EO vs. Accuracy')

    x, y = clean_graph(reports['ours']['dp'],
                    reports['ours']['accuracy'])
    plt.plot(
        x, 1. - np.array(y), '^-', color='blue', label='Ours, DP vs. Accuracy')
    x, y = clean_graph(reports['reduction']['eo'],
                       reports['reduction']['accuracy'])
    plt.plot(
        x,
        1. - np.array(y),
        '.-',
        color='green',
        label='Reduction, EO vs. Accuracy')
    x, y = clean_graph(reports['reduction']['dp'],
                       reports['reduction']['accuracy'])
    plt.plot(
        x,
        1. - np.array(y),
        '.-',
        color='black',
        label='Reduction, DP vs. Accuracy')

    # Adjust setting
    plt.xlabel('Disparity')
    plt.ylabel('Error')
    plt.legend()
    plt.title('Noise: Reductions Approach(EO) vs. Ours')
    plt.savefig('Figures/noise_compare_result.pdf')
    plt.clf()


if __name__ == '__main__':
    # Retrieve arguments from command line
    parser = argparse.ArgumentParser(description='Configurations')
    parser.add_argument('--noise_rate', type=float, default=0.1)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--lambda_value', type=float, default=0.0)
    parser.add_argument('--model', choices=['LR', 'SVM', 'MLP'], default='LR')
    parser.add_argument('--grid_size', type=int, default=20)
    parser.add_argument('--constraint', choices=['DP', 'EO'], default='DP')
    parser.add_argument('--use_all_XYA', type=bool, default=True)
    parser.add_argument('-n', type=int, default=20)
    parser.add_argument(
        '--dataset', choices=['adult', 'compas'], default='adult')
    args = parser.parse_args()
    print('Arguments: ' + str(args))
    # Run pipelines
    run_pipeline(args)
    # Plot result
    plot_result()
