import numpy as np
import pandas as pd
import json
import responsibly
from sklearn.model_selection import train_test_split
from datasets import *
from experiments import *
from utils import *
import argparse
from sklearn.cluster import KMeans

# Metrics
from fairlearn.metrics import demographic_parity_difference

# Graphing
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
plt.style.use('seaborn')

# result reports for the experiment
reports = {        
    "reweighing": {
        "accuracy": {},
        "dp": {},
        "eo": {},
        "runtime": {}
    },
    "pruning": {
        "accuracy": {},
        "dp": {},
        "eo": {},
        "runtime": {}
    },
    "unconstrained": {
        "accuracy": {},
        "dp": {},
        "eo": {},
        "runtime": {}
    },
    "Lagrangian": {
        "accuracy": {},
        "dp": {},
        "eo": {},
        "runtime": {}
    },
}


def import_dataset(dataset_name):
    """ Import compas dataset or adult data set.
    """
    print('importing dataset ' + dataset_name)
    if dataset_name == 'compas':
        X, Y, A = compas()
    elif dataset_name == 'adult_sex':
        X, Y, A = adult_sex()
    elif dataset_name == 'adult_race':
        X, Y, A = adult_race()
    elif dataset_name == 'adult_sex_race':
        X, Y, A = adult_sex_race()
    return X, Y, A


def derive_evaluation_metrics(Y_pred, Y_test, A_test):
    """ Derive accuracy, demographic parity, and equalized odds based on prediction
    results and the ground truth.
    """
    report = responsibly.fairness.metrics.report_binary(Y_test, Y_pred, A_test)
    accuracy = (Y_pred == Y_test).mean()
    dp = abs(report.loc['acceptance_rate'] - report.loc['acceptance_rate'].mean()).max()
    tpr = abs(report.loc['fnr'] - report.loc['fnr'].mean()).max()
    fpr = abs(report.loc['fpr'] - report.loc['fpr'].mean()).max()
    eo = max(tpr, fpr)
    return accuracy, dp, eo


def run_experiments(args, repeat_i, X_train, X_test, Y_train, Y_test, A_train, A_test):
    """ Run experiment pipeline to conduct traning on four different methods:
    pruning, unconstrained, reweighing, and lagrangian. Report the fairness metrics
    and the runtime.
    """
    global reports

    # Pruning
    print('Running pruning ... ')
    for num_dropped_features in range(args.n):
        Y_pred, run_time = pruning(num_dropped_features, args.constraint, args.model, args.lambda_value, X_train, X_test, Y_train, A_train)
        accuracy, dp, eo = derive_evaluation_metrics(Y_pred, Y_test, A_test)
        if num_dropped_features not in reports["pruning"]["accuracy"]:
            reports["pruning"]["accuracy"][num_dropped_features] = []
            reports["pruning"]["dp"][num_dropped_features] = []
            reports["pruning"]["eo"][num_dropped_features] = []
            reports["pruning"]["runtime"][num_dropped_features] = []
        reports["pruning"]["accuracy"][num_dropped_features].append(accuracy)
        reports["pruning"]["dp"][num_dropped_features].append(dp)
        reports["pruning"]["eo"][num_dropped_features].append(eo) 
        reports["pruning"]["runtime"][num_dropped_features].append(run_time) 
    
    # Unconstrained
    print('Running unconstrained ... ')
    Y_pred, run_time = unconstraint(args.model, X_train, X_test, A_train, A_test, Y_train)
    accuracy, dp, eo = derive_evaluation_metrics(Y_pred, Y_test, A_test)
    if 0 not in reports["unconstrained"]["accuracy"]:
        reports["unconstrained"]["accuracy"][0] = []
        reports["unconstrained"]["dp"][0] = []
        reports["unconstrained"]["eo"][0] = []
        reports["unconstrained"]["runtime"][0] = []
    reports["unconstrained"]["accuracy"][0].append(accuracy)
    reports["unconstrained"]["dp"][0].append(dp)
    reports["unconstrained"]["eo"][0].append(eo)
    reports["unconstrained"]["runtime"][0].append(run_time)

    # Reweighing
    print('Running reweighing ... ')
    Y_pred, run_time = reweighing(args.model, X_train, A_train, Y_train, X_test)
    accuracy, dp, eo = derive_evaluation_metrics(Y_pred, Y_test, A_test)
    if 0 not in reports["reweighing"]["accuracy"]:
        reports["reweighing"]["accuracy"][0] = []
        reports["reweighing"]["dp"][0] = []
        reports["reweighing"]["eo"][0] = []
        reports["reweighing"]["runtime"][0] = []
    reports["reweighing"]["accuracy"][0].append(accuracy)
    reports["reweighing"]["dp"][0].append(dp)
    reports["reweighing"]["eo"][0].append(eo)
    reports["reweighing"]["runtime"][0].append(run_time)

    # Lagrangian
    print('Running lagrangian ... ')
    constraint_weights = np.arange(args.partitions + 1) / args.partitions 
    for constraint_weight in constraint_weights:
        Y_pred, run_time = lagrangian(args.constraint, args.model, constraint_weight, args.grid_size, X_train, Y_train, A_train, X_test)
        accuracy, dp, eo = derive_evaluation_metrics(Y_pred, Y_test, A_test)
        if constraint_weight not in reports["Lagrangian"]["accuracy"]:
            reports["Lagrangian"]["accuracy"][constraint_weight] = []
            reports["Lagrangian"]["dp"][constraint_weight] = []
            reports["Lagrangian"]["eo"][constraint_weight] = []
            reports["Lagrangian"]["runtime"][constraint_weight] = []
        reports["Lagrangian"]["accuracy"][constraint_weight].append(accuracy)
        reports["Lagrangian"]["dp"][constraint_weight].append(dp)
        reports["Lagrangian"]["eo"][constraint_weight].append(eo)
        reports["Lagrangian"]["runtime"][constraint_weight].append(run_time)


def run_on_celeba(args):
    """ Retrieve CelebA dataset and conduct experiment.
    """
    if args.dataset == 'celeba':
        X_train, X_test, Y_train, Y_test, A_train, A_test = celeba()
    elif args.dataset == 'celeba_young_male':
        X_train, X_test, Y_train, Y_test, A_train, A_test = celeba_young_male()
    if args.noise_rate > 0 and args.asymmetric_noise is False:
        A_train = add_noise(args.noise_rate, A_train)
    if args.noise_rate > 0 and args.asymmetric_noise:
        A_train = add_asymmetric_noise(args.noise_rate, A_train)
    run_experiments(args, 0, X_train, X_test, Y_train, Y_test, A_train, A_test)


def run_on_other_datasets(args):
    """ Retrieve compas dataset or adult dataset, and conduct experiment.
    """
    X, Y, A = import_dataset(args.dataset)
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
        if args.noise_rate > 0. and args.asymmetric_noise is False:
            A_train = add_noise(args.noise_rate, A_train)
        if args.noise_rate > 0. and args.asymmetric_noise:
            A_train = add_asymmetric_noise(args.noise_rate, A_train)
        run_experiments(args, i, X_train, X_test, Y_train, Y_test, A_train, A_test)


def run_pipeline(args):
    """ Run the experiment pipeline and report the results.
    """
    if args.dataset in ['celeba', 'celeba_young_male']:
        run_on_celeba(args)
    else:
        run_on_other_datasets(args)
    print('Post-processing ... ')
    post_processing()
    print('output results ... ')
    with open('results/results_' + str(args) + '.json',
              'w') as fp:
        json.dump(reports, fp)


def post_processing():
    """ Process reports to take mean value of each attribute.
    """
    for experiment in reports:
        for matrix in reports[experiment]:
            res = []
            for k in reports[experiment][matrix]:
                res.append(np.array(reports[experiment][matrix][k]).mean())
            reports[experiment][matrix] = res


def plot_result(args):
    """Plot results as test error vs. fairness violation.

    For CelebA dataset, take 15 partitions of accuracy, select the left 
    most point so that the figure won't be too crowd.
    """
    print('printing ' + str(args) + '...')
    plt.clf()
    with open(
            'results/results_' + str(args) + '.json',
            'r') as fp:
        reports = json.load(fp)
    palette = sns.color_palette() 

    markers = ['o', 'o', 'o', 'o', 'o']
    count = 0
    
    for attribute in ['eo', 'dp']:
        if attribute != args.constraint.lower():
            continue
        for experiment in ['pruning', 'reweighing', 'unconstrained', 'Lagrangian']:
            
            if experiment != 'pruning' or args.dataset not in ['celeba', 'celeba_young_male']:
                x, y = reports[experiment][attribute], reports[experiment]['accuracy']
                
            else:
                n = 15
                points = [[] for _ in range(n+1)]
                x, y = reports[experiment][attribute], reports[experiment]['accuracy']
                y_min, y_max = min(y), max(y)

                slice_range = 1. * (y_max - y_min) / n
                for i in range(len(reports[experiment][attribute])):
                    if 1 - reports[experiment]['accuracy'][i] > 0.115 and args.constraint == 'EO':
                        continue
                    points[int((reports[experiment]['accuracy'][i]-y_min)/slice_range)].append([reports[experiment][attribute][i], reports[experiment]['accuracy'][i]])
                x, y = [], []
                for label in range(len(points)):
                    points[label].sort()
                    if len(points[label]) != 0:
                        x.append(points[label][0][0])
                        y.append(points[label][0][1])

            if experiment != 'Lagrangian':
                plt.plot(x, 1. - np.array(y), markers[count%(len(reports))], color=palette[count%(len(reports))], label=experiment, markersize=14)
            else:
                plt.plot(x, 1. - np.array(y), markers[count%(len(reports))], color='orange', label=experiment, markersize=14)
            count += 1

        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlabel('Max Violation', fontsize=18)
        plt.ylabel('Error', fontsize=18)
        plt.legend(fontsize=18)
        plt.savefig('figures/figures_'+attribute+str(args)+'.pdf', bbox_inches='tight')
        plt.clf()


if __name__ == '__main__':
    # Retrieve arguments from command line
    parser = argparse.ArgumentParser(description='Configurations')
    parser.add_argument('--noise_rate', type=float, default=0.0)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--grid_size', type=int, default=5)
    parser.add_argument('--asymmetric_noise', type=bool, default=False)
    parser.add_argument('--lambda_value', type=float, default=0.0)
    parser.add_argument('--model', choices=['LR', 'SVM', 'MLP'], default='LR')
    parser.add_argument('--constraint', choices=['DP', 'EO'], default='DP')
    parser.add_argument('-n', type=int, default=20)
    parser.add_argument('--partitions', type=int, default=5)
    parser.add_argument(
        '--dataset', choices=['adult_sex', 'adult_race', 'adult_sex_race', 'compas', 'celeba', 'celeba_young_male'], default='adult_sex')
    args = parser.parse_args()
    print('Arguments: ' + str(args))
    # Run pipelines
    run_pipeline(args)
    # Plot results
    plot_result(args)
