import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

with open('results/CelebA.json', 'r') as fp:
    reports = json.load(fp)

n = len(reports['accuracy'])

palette = sns.color_palette()
fig, ax1 = plt.subplots()
fig.subplots_adjust(right=0.75)
ax2 = ax1.twinx()
ax2.grid(b=False)

ax1.set_xlabel('# of Blocked Features')
ax1.set_ylabel('Mutual Information')
ax1.set_ylabel('Accuracy')
ax2.set_ylabel('Disparity')
ax1.set_xlim(0, n)
ax2.axis('on')
idx = np.arange(n)
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
plt.savefig(f'Figures/CelebA_Age.pdf')

plt.clf()

pareto = [(reports['accuracy'][i], reports['DP'][i]) for i in range(argsn)]
pareto = sorted(pareto)

x = [pareto[0][0]]
y = [pareto[0][1]]
for i in range(1, argsn):
    if pareto[i][1] >= y[-1]:
        x.append(pareto[i][0])
        y.append(pareto[i][1])

# plt.scatter(reports['accuracy'], reports['DP'])
plt.plot(1-x, y, 'bo-')
plt.xlabel('Error')
plt.ylabel('Fairness')
plt.title('CelebA Dataset')
plt.show()