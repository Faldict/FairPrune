import numpy as np
import pandas as pd
import json
import responsibly
from sklearn.model_selection import train_test_split
from fairlearn.reductions import GridSearch, ErrorRate
from fairlearn.reductions import DemographicParity, EqualizedOdds
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

import argparse
from utils import equal_opportunity_difference
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
plt.style.use('seaborn')

parser = argparse.ArgumentParser(description='Configurations')
parser.add_argument('--repeats', type=int, default=5)
parser.add_argument('--grid_size', type=int, default=10)
parser.add_argument('--dataset', choices=['adult', 'compas'], default='adult')
parser.add_argument('--model', choices=['LR', 'SVM', 'MLP'], default='LR')
parser.add_argument('--constraint', choices=['DP', 'EO'], default='DP')
parser.add_argument('--partitions', type=int, default=10)
args = parser.parse_args()

# import dataset
from datasets import *
if args.dataset == 'compas':
    X, Y, A = compas()
else:
    X, Y, A = adult()

# import models
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
models = {
    'LR': LogisticRegression(solver='liblinear', fit_intercept=True),
    'SVM': SVC(gamma='auto'),
    'MLP': MLPClassifier(
        hidden_layer_sizes=(100,), random_state=1, max_iter=300)
}


# pipeline
def run(X, Y, A, repeats=5):
    reports = {
        'accuracy': [],
        'DP': [],
        'EO': [],
    }
    constraint_weights = np.arange(args.partitions + 1) / args.partitions / 2.

    for constraint_weight in constraint_weights:
        print(f'Running for constraint weight={constraint_weight}')
        accuracy = []
        dp, equal_tpr, equal_fpr, eo = [], [], [], []
        for i in range(repeats):
            print(f'Running pipeline #{i+1} out of {repeats}')
            X_train, X_test, Y_train, Y_test, A_train, A_test = train_test_split(
                X,
                Y,
                A,
                test_size=0.2,
                random_state=131 + i,
                stratify=np.stack((Y, A), axis=1))
            X_train = X_train.reset_index(drop=True)
            X_test = X_test.reset_index(drop=True)
            if args.constraint == 'DP':
                clf = GridSearch(
                    models[args.model],
                    constraints=DemographicParity(),
                    constraint_weight=constraint_weight,
                    grid_size=args.grid_size)
            elif args.constraint == 'EO':
                clf = GridSearch(
                    models[args.model],
                    constraints=EqualizedOdds(),
                    constraint_weight=constraint_weight,
                    grid_size=args.grid_size)
            print('Fitting ...')
            clf.fit(X_train, Y_train, sensitive_features=A_train)
            print('Predicting ...')
            Y_pred = clf.predict(X_test)

            print('Calculating accuracy, DP, and EO ...')
            accuracy.append((Y_pred == Y_test).mean())
            dp.append(
                demographic_parity_difference(
                    y_true=Y_test, y_pred=Y_pred, sensitive_features=A_test))
            eo.append(
                equal_opportunity_difference(
                    Y_true=Y_test, Y_pred=Y_pred, sensitive_features=A_test))

        print('Generating reports ... ')
        print(
            np.array(eo).mean(),
            np.array(dp).mean(),
            np.array(accuracy).mean())
        reports['accuracy'].append(np.array(accuracy).mean())
        reports['DP'].append(np.array(dp).mean())
        reports['EO'].append(np.array(eo).mean())
    return reports


reports = run(X, Y, A, args.repeats)

with open(
        f'results/grid_search_{args.dataset}_{args.model}_{args.constraint}_{args.grid_size}_{args.partitions}.json',
        'w') as fp:
    json.dump(reports, fp)
