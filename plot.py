import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

with open('results/CelebA_00.json', 'r') as fp:
    report1 = json.load(fp)

with open('results/CelebA_01.json', 'r') as fp:
    report2 = json.load(fp)

with open('results/CelebA_05.json', 'r') as fp:
    report3 = json.load(fp)

n = np.arange(len(report1['accuracy'])) * 10

palette = sns.color_palette()
fig, ax = plt.subplots()
ax.plot(n, np.array(report1['DP']), color=palette[0], label='random', linewidth=8, linestyle='dashed')
ax.plot(n, np.array(report2['DP']), color=palette[1], label='ascending', linewidth=8, linestyle='dashed')
ax.plot(n, np.array(report3['DP']), color=palette[2], label='descending', linewidth=8, linestyle='dashed')
# ax.legend(loc=3, fontsize=24)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('# of Dropped Channels', fontsize=24)
plt.ylabel('Max Violation', fontsize=24)
plt.savefig('Figures/CelebA_lambda_dp.pdf', bbox_inches='tight')

plt.clf()
fig, ax = plt.subplots()
ax.plot(n, np.array(report1['EO']), color=palette[0], label='random', linewidth=8, linestyle='dashed')
ax.plot(n, np.array(report2['EO']), color=palette[1], label='ascending', linewidth=8, linestyle='dashed')
ax.plot(n, np.array(report3['EO']), color=palette[2], label='descending', linewidth=8, linestyle='dashed')
# ax.legend(loc=3, fontsize=24)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('# of Dropped Channels', fontsize=24)
plt.ylabel('Max Violation', fontsize=24)
plt.savefig('Figures/CelebA_lambda_eo.pdf', bbox_inches='tight')

plt.clf()
fig, ax = plt.subplots()
ax.plot(n, np.array(report1['accuracy']), color=palette[0], label=r'$\lambda = 0.0$', linewidth=8, linestyle='dashed')
ax.plot(n, np.array(report2['accuracy']), color=palette[1], label=r'$\lambda = 0.1$', linewidth=8, linestyle='dashed')
ax.plot(n, np.array(report3['accuracy']), color=palette[2], label=r'$\lambda = 0.5$', linewidth=8, linestyle='dashed')
# ax.legend(loc=3, fontsize=24)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('# of Dropped Channels', fontsize=24)
plt.ylabel('Accuracy', fontsize=24)
plt.savefig('Figures/CelebA_lambda_accuracy.pdf', bbox_inches='tight')

figsize = (2, 2)
fig_leg = plt.figure(figsize=figsize)
ax_leg = fig_leg.add_subplot(111)
# add the legend from the previous axes
ax_leg.legend(*ax.get_legend_handles_labels(), loc='center', fontsize=12)
ax_leg.axis('off')
fig_leg.savefig('Figures/CelebA_lambda_legend.pdf', bbox_inches='tight')