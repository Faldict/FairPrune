import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set()
plt.style.use('seaborn')

# Read report files
with open(f'results/grid_search_adult_LR_DP_15_12.json', 'r') as fp:
    dp_reports = json.load(fp)

with open(f'results/grid_search_adult_LR_EO_15_12.json', 'r') as fp:
    eo_reports = json.load(fp)

with open(f'results/main.json', 'r') as fp:
    main_reports = json.load(fp)


# Clean the graph of our experiment
def clean_graph(accuracy, disparity):
    pareto = [(accuracy[i], disparity[i]) for i in range(len(disparity))]
    pareto = sorted(pareto)

    x = [pareto[0][0]]
    y = [pareto[0][1]]
    for i in range(1, len(disparity)):
        if pareto[i][1] >= y[-1]:
            x.append(pareto[i][0])
            y.append(pareto[i][1])
    return x, y


# Plot the comparison of DP.
plt.plot(
    dp_reports['DP'],
    1 - np.array(dp_reports['accuracy']),
    '^-',
    color='red',
    label='GridSearch(DP)')
main_dp_accuracy, main_dp = clean_graph(main_reports['accuracy'],
                                        main_reports['DP'])
plt.plot(
    main_dp,
    1 - np.array(main_dp_accuracy),
    'o-',
    color='blue',
    label='Our Experiment')
plt.xlabel('Disparity(Demographic Parity)')
plt.ylabel('Error')
plt.legend()
plt.title('GridSearch(DP) vs. Our Experiment')
plt.savefig('Figures/dp_main_compare.pdf')

plt.clf()

# Plot the comparison of EO
plt.plot(
    eo_reports['EO'],
    1 - np.array(eo_reports['accuracy']),
    '^-',
    color='red',
    label='GridSearch(EO)')
main_eo_accuracy, main_eo = clean_graph(main_reports['accuracy'],
                                        main_reports['EO'])
plt.plot(
    main_eo,
    1 - np.array(main_eo_accuracy),
    'o-',
    color='blue',
    label='Our Experiment')
plt.xlabel('Disparity(Equal Opportunity)')
plt.ylabel('Error')
plt.legend()
plt.title('GridSearch(EO) vs. Our Experiment')
plt.savefig('Figures/eo_main_compare.pdf')
