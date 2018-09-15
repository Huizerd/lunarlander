import json

import matplotlib.pyplot as plt
import numpy as np

# Matplotlib config
plt.style.use('ggplot')
# plt.rcParams['lines.linewidth'] = 2
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 14

# Sarsa
with open('record/sarsa_bins3/grid_search.json', 'r') as data_file:
    sarsa = json.loads(data_file.read())

# Hyperparameters and their config numbers (low, default, high)
sa_hyperparam = [('bins', 1, 2, 3, [5, 6, 7]),
                 ('epsilon steps', 1, 2, 3, [5e4, 1e5, 1.5e5]),
                 ('epsilon decay', 18, 6, 30, [0.9, 0.97, 0.999]),
                 ('alpha steps', 1, 2, 3, [3e5, 4e5, 5e5]),
                 ('alpha decay', 1, 2, 3, [0.97, 1, np.nan])]

# Go over all and create plots
fig, axes = plt.subplots(2, 3, figsize=(16, 8))  # , sharex='all', sharey='all')

for hp, ax in zip(sa_hyperparam, axes.flat):
    episode = [sarsa['scores'][hp[1]][1], sarsa['scores'][hp[2]][1], sarsa['scores'][hp[3]][1]]
    score = [sarsa['scores'][hp[1]][2], sarsa['scores'][hp[2]][2], sarsa['scores'][hp[3]][2]]

    episode_p = []
    score_p = []
    hyperparam_p = []

    for j in range(len(episode)):
        episode_p.append((episode[j] - episode[1]) / episode[j] * 100)
        score_p.append((score[j] - score[1]) / score[j] * 100)
        hyperparam_p.append((hp[4][j] - hp[4][1]) / hp[4][j] * 100)

    ax.plot(hyperparam_p, score_p, label='max score')
    ax.plot(hyperparam_p, episode_p, label='episodes for max score')
    ax.set_title(f'SA: {hp[0]}', fontstyle='italic')
    ax.set_xlabel('hyperparameter change [%]')
    ax.set_ylabel('performance change [%]')
    ax.legend()

# Delete last subplot
axes[-1, -1].axis('off')

fig.tight_layout()
plt.show()

# config = [18, 6, 30]
# episode = [data['scores'][18][1], data['scores'][6][1], data['scores'][30][1]]
# score = [data['scores'][18][2], data['scores'][6][2], data['scores'][30][2]]
# eps = [0.9, 0.97, 0.999]

# for i in data['scores']:
#     config.append(i[0])
#     episode.append(i[1])
#     score.append(i[2])
#
# plt.plot(episode, score, 'b*')
#
# ax = plt.gca()
# for i, txt in enumerate(config):
#     ax.annotate(txt, (episode[i], score[i]))
#
# plt.show()

# episode_p = []
# score_p = []
# eps_p = []
#
#
#
# for i in range(len(config)):
#     eps_p.append((eps[i] - eps[1]) / eps[i] * 100)
#     episode_p.append((episode[i] - episode[1]) / episode[i] * 100)
#     score_p.append((score[i] - score[1]) / score[i] * 100)
#
# print(episode)
# print(score)
#
#
# plt.plot(eps_p, score_p, label='max score')
# plt.plot(eps_p, episode_p, label='episodes for max score')
# ax = plt.gca()
# fig = plt.gcf()
# fig.set_facecolor('w')
# ax.set_facecolor('w')
#
# plt.title('Sensitivity analysis: epsilon decay', fontstyle='italic')
# plt.xlabel('change in hyperparameter [%]')
# plt.ylabel('change in performance [%]')
# plt.legend()
#
# plt.tight_layout()
# plt.show()
