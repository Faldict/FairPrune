import numpy as np
import pandas as pd
import json
import responsibly
from sklearn.model_selection import train_test_split
from datasets import *
from experiments import *
from utils import *
import argparse

# Graphing
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
plt.style.use('seaborn')


def runtime_result_analysis(args):
    """ Calcualte the mean, max, min runtime for each experiment.
    """
    print('printing ' + str(args) + '...')
    with open('results/results_' + str(args) + '.json', 'r') as fp:
        reports = json.load(fp)
    
    for experiment in ['pruning', 'unconstrained', 'reweighing', 'Lagrangian']:
        runtime = reports[experiment]['runtime']
        max_runtime = max(runtime)
        min_runtime = min(runtime)
        average_runtime = np.array(runtime).mean()
        print('-------------------------------')
        print(experiment)
        print('Maximum runtime: ' + str(max_runtime))
        print('Minimum runtime: ' + str(min_runtime))
        print('Average runtime: ' + str(average_runtime))


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
        '--dataset', choices=['adult_sex', 'adult_race', 'adult_sex_race', 'compas', 'celeba'], default='adult_sex')
    args = parser.parse_args()
    print('Arguments: ' + str(args))
    runtime_result_analysis(args)