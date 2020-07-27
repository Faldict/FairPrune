import numpy as np
import pandas as pd
import json
import responsibly
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import argparse
from utils import mutual_information_2d
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

argsn = 128
X_train = np.load('celeba.train.npy')
X_test = np.load('celeba.test.npy')
target_train = np.load('celeba_label.train.npy')
target_test = np.load('celeba_label.test.npy')
Y_train = target_train[:, 31]
Y_test = target_test[:, 31]
A_train = target_train[:, 20]
A_test = target_test[:, 20]

def run(X_train, Y_train, A_train, X_test, Y_test, A_test, repeats=5):
    accuracy = []
    dp, equal_tpr, equal_fpr, eo = [], [], [], []
    mi = []

    for i in range(repeats):
        clf = LogisticRegression(solver='liblinear', fit_intercept=True)
        clf.fit(X_train, Y_train)
        Y_pred = clf.predict(X_test)

        report = responsibly.fairness.metrics.report_binary(Y_test, Y_pred, A_test)
        accuracy.append((Y_pred == Y_test).mean())
        mi.append(mutual_information_2d(Y_pred, A_test))
        dp.append(abs(report.loc['acceptance_rate'] - report.loc['acceptance_rate'].mean()).max())
        tpr = abs(report.loc['fnr'] - report.loc['fnr'].mean()).max()
        fpr = abs(report.loc['fpr'] - report.loc['fpr'].mean()).max()
        equal_tpr.append(tpr)
        equal_fpr.append(fpr)
        eo.append(max(tpr, fpr))

    reports = {
            'accuracy' : np.array(accuracy).mean(),
            'accuracy_std' : np.array(accuracy).std(),
            'DP' : np.array(dp).mean(),
            'DP_std' : np.array(dp).std(),
            'TPR' : np.array(tpr).mean(),
            'TPR_std' : np.array(tpr).std(),
            'FPR' : np.array(fpr).mean(),
            'FPR_std' : np.array(fpr).std(),
            'EO' : np.array(eo).mean(),
            'EO_std' : np.array(eo).std(),
            'MI' : np.array(mi).mean(),
            'MI_std' : np.array(mi).std()
    }
    return reports

mis = []
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape, A_train.shape, A_test.shape)
for col in range(X_train.shape[1]):
    mi = mutual_information_2d(X_train[(Y_train == 1), col].squeeze(), A_train[Y_train == 1]) + mutual_information_2d(X_train[(Y_train == 0), col].squeeze(), A_train[Y_train == 0])
    mis.append((mi, col))
mis = sorted(mis, reverse=False)
mis1 = [l[1] for l in mis]

reports = {'accuracy':[], 'accuracy_std':[], 'DP':[], 'DP_std':[], 'TPR':[], 'TPR_std':[], 'FPR': [], 'FPR_std':[], 'EO': [], 'EO_std': [], 'MI': [], 'MI_std': []}
for i in range(argsn):
    Xt_train = X_train[:, mis1[:X_train.shape[1]-i]]
    Xt_test = X_test[:, mis1[:X_test.shape[1]-i]]
    report = run(Xt_train, Y_train, A_train, Xt_test, Y_test, A_test, repeats=1)
    for k in reports.keys():
        reports[k].append(report[k])
    if i % 5 == 0:
        print(f'[INFO] Block {i} features.')
with open('results/CelebA_Sex_EO.json', 'w') as fp:
    json.dump(reports, fp)

palette = sns.color_palette()
fig, ax1 = plt.subplots()
fig.subplots_adjust(right=0.75)
ax2 = ax1.twinx()
ax2.grid(b=False)

ax1.set_xlabel('# of Blocked Features')
ax1.set_ylabel('Mutual Information')
ax1.set_ylabel('Accuracy')
ax2.set_ylabel('Disparity')
ax1.set_xlim(0, argsn)
ax2.axis('on')
idx = np.arange(argsn)
p1, = ax2.plot(idx, reports['DP'], color=palette[2], label='DP', marker='+')
# ax2.fill_between(idx, reports['DP']-reports['DP_std'], reports['DP']+reports['DP_std'], color=palette[2], alpha=0.5)
p2, = ax2.plot(idx, reports['EO'], color=palette[3], label='EO')
# par2.fill_between(idx, reports['EO']-reports['EO_std'], reports['EO']+reports['EO_std'], color=palette[3], alpha=0.5)
ax2.tick_params(axis='y', labelcolor=palette[2])

p3, = ax1.plot(idx, reports['accuracy'], color=palette[0], label='Accuracy')
# par1.fill_between(idx, reports['accuracy']-reports['accuracy_std'], reports['accuracy']+reports['accuracy_std'], color=palette[0], alpha=0.5)
ax1.tick_params(axis='y', labelcolor=palette[0])

lines = [p1, p2, p3]
ax1.legend(lines, [l.get_label() for l in lines])
ax1.set_title(f'CelebA', fontsize=24)
plt.savefig(f'Figures/CelebA_Sex_EO .pdf')

plt.clf()

pareto = [(reports['accuracy'][i], reports['EO'][i]) for i in range(argsn)]
pareto = sorted(pareto)

x = [pareto[0][0]]
y = [pareto[0][1]]
for i in range(1, argsn):
    if pareto[i][1] >= y[-1]:
        x.append(pareto[i][0])
        y.append(pareto[i][1])
print(x)
print(y)
# plt.scatter(reports['accuracy'], reports['DP'])
plt.plot(x, y, 'bo-')
plt.xlabel('Accuracy')
plt.ylabel('Fairness')
plt.title('CelebA Dataset')
plt.show()
