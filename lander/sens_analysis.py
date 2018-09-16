import json

import matplotlib.pyplot as plt

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
data_files = ['record/final/sarsa_bins/grid_search.json', 'record/final/sarsa_lr/grid_search.json',
              'record/final/sarsa_greedy/grid_search.json']
scores = []

for f in data_files:
    with open(f, 'r') as data_file:
        scores.append(json.loads(data_file.read()))

# Hyperparameters: parameter name, file number, config numbers, default config number and parameter values
sa_hyperparam = [('bins', 0, [0, 1, 2, 3, 4], 2, [3, 4, 5, 6, 7]),
                 ('alpha', 1, [0, 1, 2, 3, 4], 2, [0.1, 0.15, 0.2, 0.25, 0.3]),
                 ('epsilon steps', 2, [4, 0, 3], 1, [5e4, 1e5, 1.5e5]),
                 ('epsilon decay', 2, [1, 0, 2], 1, [0.9, 0.97, 0.999])]

# Go over all and create plots
fig, axes = plt.subplots(2, 2, figsize=(10, 10))  # , sharex='all', sharey='all')

for hp, ax in zip(sa_hyperparam, axes.flat):

    episode = []
    score = []

    for i in range(len(hp[2])):
        episode.append(scores[hp[1]]['scores'][hp[2][i]][1])
        score.append(scores[hp[1]]['scores'][hp[2][i]][2])

    episode_p = []
    score_p = []
    hyperparam_p = []

    for j in range(len(episode)):
        episode_p.append((episode[j] - episode[hp[3]]) / episode[hp[3]] * 100)
        score_p.append((score[j] - score[hp[3]]) / score[hp[3]] * 100)
        hyperparam_p.append((hp[4][j] - hp[4][hp[3]]) / hp[4][hp[3]] * 100)

    ax.plot(hyperparam_p, score_p, label='max avg score')
    ax.plot(hyperparam_p, episode_p, label='episodes for max avg score')
    ax.set_title(f'SA: {hp[0]}', fontstyle='italic')
    ax.set_xlabel('hyperparameter change [%]')
    ax.set_ylabel('performance change [%]')
    ax.legend()

fig.tight_layout()
fig.savefig('figure/sens_sarsa.pdf')
plt.show()

# Qlearn
data_files = ['record/final/qlearn_bins/grid_search.json', 'record/final/qlearn_lr/grid_search.json',
              'record/final/qlearn_greedy/grid_search.json']
scores = []

for f in data_files:
    with open(f, 'r') as data_file:
        scores.append(json.loads(data_file.read()))

# Hyperparameters: parameter name, file number, config numbers, default config number and parameter values
sa_hyperparam = [('bins', 0, [0, 1, 2, 3, 4], 2, [3, 4, 5, 6, 7]),
                 ('alpha', 1, [0, 1, 2, 3, 4], 2, [0.1, 0.15, 0.2, 0.25, 0.3]),
                 ('epsilon steps', 2, [4, 0, 3], 1, [5e4, 1e5, 1.5e5]),
                 ('epsilon decay', 2, [1, 0, 2], 1, [0.9, 0.97, 0.999])]

# Go over all and create plots
fig, axes = plt.subplots(2, 2, figsize=(10, 10))  # , sharex='all', sharey='all')

for hp, ax in zip(sa_hyperparam, axes.flat):

    episode = []
    score = []

    for i in range(len(hp[2])):
        episode.append(scores[hp[1]]['scores'][hp[2][i]][1])
        score.append(scores[hp[1]]['scores'][hp[2][i]][2])

    episode_p = []
    score_p = []
    hyperparam_p = []

    for j in range(len(episode)):
        episode_p.append((episode[j] - episode[hp[3]]) / episode[hp[3]] * 100)
        score_p.append((score[j] - score[hp[3]]) / score[hp[3]] * 100)
        hyperparam_p.append((hp[4][j] - hp[4][hp[3]]) / hp[4][hp[3]] * 100)

    ax.plot(hyperparam_p, score_p, label='max avg score')
    ax.plot(hyperparam_p, episode_p, label='episodes for max avg score')
    ax.set_title(f'SA: {hp[0]}', fontstyle='italic')
    ax.set_xlabel('hyperparameter change [%]')
    ax.set_ylabel('performance change [%]')
    ax.legend()

fig.tight_layout()
fig.savefig('figure/sens_qlearn.pdf')
plt.show()

# Doubledqn
data_files = ['record/final/doubledqn_batchsize/grid_search.json', 'record/final/doubledqn_layersize/grid_search.json']
scores = []

for f in data_files:
    with open(f, 'r') as data_file:
        scores.append(json.loads(data_file.read()))

# Hyperparameters: parameter name, file number, config numbers, default config number and parameter values
sa_hyperparam = [('batch size', 0, [0, 1, 2], 1, [32, 64, 128]),
                 ('layer size', 1, [0, 1, 2], 1, [256, 512, 1024])]

# Go over all and create plots
fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # , sharex='all', sharey='all')

for hp, ax in zip(sa_hyperparam, axes.flat):

    episode = []
    score = []

    for i in range(len(hp[2])):
        episode.append(scores[hp[1]]['scores'][hp[2][i]][1])
        score.append(scores[hp[1]]['scores'][hp[2][i]][2])

    episode_p = []
    score_p = []
    hyperparam_p = []

    for j in range(len(episode)):
        episode_p.append((episode[j] - episode[hp[3]]) / episode[hp[3]] * 100)
        score_p.append((score[j] - score[hp[3]]) / score[hp[3]] * 100)
        hyperparam_p.append((hp[4][j] - hp[4][hp[3]]) / hp[4][hp[3]] * 100)

    ax.plot(hyperparam_p, score_p, label='max avg score')
    ax.plot(hyperparam_p, episode_p, label='episodes for max avg score')
    ax.set_title(f'SA: {hp[0]}', fontstyle='italic')
    ax.set_xlabel('hyperparameter change [%]')
    ax.set_ylabel('performance change [%]')
    ax.legend()

fig.tight_layout()
# fig.savefig('figure/sens_doubledqn.pdf')
plt.show()
