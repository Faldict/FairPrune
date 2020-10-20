import numpy as np
import pandas as pd
import json
import responsibly
from sklearn.model_selection import train_test_split

import argparse
from utils import mutual_information_2d
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
plt.style.use('seaborn')

parser = argparse.ArgumentParser(description='Configurations')
parser.add_argument('-n', type=int, default=10)
parser.add_argument('--repeats', type=int, default=5)
parser.add_argument('--dataset', choices=['adult', 'compas', 'crime'], default='adult')
parser.add_argument('--model', choices=['LR', 'SVM', 'MLP'], default='LR')
parser.add_argument('--constraint', choices=['DP', 'EO'], default='DP')
parser.add_argument('--lbda', type=float, default=0.0)
args = parser.parse_args()

# import dataset
from datasets import *
if args.dataset == 'compas':
    X, Y, A = compas()
elif args.dataset == 'crime':
    X, Y, A = crime()
    # A = np.digitize(A, np.array([0.05, 0.10, 0.3]))
else:
    X, Y, A = adult()

print(Y)

# import models
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
models = {
    'LR' : LogisticRegression(solver='liblinear', fit_intercept=True),
    'SVM' : SVC(gamma='auto'),
    'MLP' : MLPClassifier(hidden_layer_sizes=(100,), random_state=1, max_iter=300)
}

# pipeline
def run(X, Y, A, repeats=5):
    accuracy = []
    dp, equal_tpr, equal_fpr, eo = [], [], [], []
    mi = []
    for i in range(repeats):

        X_train, X_test, Y_train, Y_test, A_train, A_test = train_test_split(X, 
                                                        Y, 
                                                        A,
                                                        test_size = 0.2,
                                                        random_state=131+i)
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
  
        clf = models[args.model]
        clf.fit(X_train, Y_train)

        Y_pred = clf.predict(X_test)

        # report = responsibly.fairness.metrics.report_binary(Y_test, Y_pred, A_test)
        accuracy.append((Y_pred == Y_test).mean())
        mi.append(mutual_information_2d(Y_pred, A_test))
        # dp.append(abs(report.loc['acceptance_rate'] - report.loc['acceptance_rate'].mean()).max())
        # tpr = abs(report.loc['fnr'] - report.loc['fnr'].mean()).max()
        # fpr = abs(report.loc['fpr'] - report.loc['fpr'].mean()).max()
        # equal_tpr.append(tpr)
        # equal_fpr.append(fpr)
        # eo.append(max(tpr, fpr))

    reports = {
            'accuracy' : np.array(accuracy).mean(),
            'accuracy_std' : np.array(accuracy).std(),
            # 'DP' : np.array(dp).mean(),
            # 'DP_std' : np.array(dp).std(),
            # 'TPR' : np.array(tpr).mean(),
            # 'TPR_std' : np.array(tpr).std(),
            # 'FPR' : np.array(fpr).mean(),
            # 'FPR_std' : np.array(fpr).std(),
            # 'EO' : np.array(eo).mean(),
            # 'EO_std' : np.array(eo).std(),
            'MI' : np.array(mi).mean(),
            'MI_std' : np.array(mi).std()
    }
    return reports

# compute mutual information between X and A
mis = []
for col in X.columns:
    if args.constraint == 'EO':
        mi = mutual_information_2d(X[col].values[Y==0], A[Y==0]) + mutual_information_2d(X[col].values[Y==1], A[Y==1])
    else:
        mi = mutual_information_2d(X[col].values, A) - args.lbda * mutual_information_2d(X[col].values, Y)
    mis.append((mi, col))
mis = sorted(mis, reverse=False)
mis1 = [l[1] for l in mis]
reports = {'accuracy':[], 'accuracy_std':[], 'MI': [], 'MI_std': []}
for i in range(args.n):
    Xt = X[mis1[:len(X.columns)-i]]
    report = run(Xt, Y, A, repeats=args.repeats)
    for k in reports.keys():
        reports[k].append(report[k])
    if i % 5 == 0:
        print(f'[INFO] Block {i} features.')
for k in reports.keys():
    reports[k] = np.array(reports[k])

# with open(f'{args.dataset}_{args.model}_{args.constraint}_Drop{args.n}_{args.lbda}.json', 'w') as fp:
#     json.dump(reports, fp)

# plot

plt.plot(np.arange(args.n), reports['accuracy'])
plt.fill_between(np.arange(args.n), reports['accuracy'] - reports['accuracy_std'], reports['accuracy'] + reports['accuracy_std'], alpha=0.3)
plt.xlabel("# of Dropped Features", fontsize=20)
plt.ylabel("Accuracy", fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig("Figures/crime_accuracy.pdf", bbox_inches='tight')

plt.clf()
plt.plot(np.arange(args.n), reports['MI'])
plt.fill_between(np.arange(args.n), reports['MI'] - reports['MI_std'], reports['MI'] + reports['MI_std'], alpha=0.3)
plt.xlabel("# of Dropped Features", fontsize=20)
plt.ylabel("Mutual Information", fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig("Figures/crime_mi.pdf", bbox_inches='tight')

