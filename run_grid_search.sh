#!/usr/bin/env bash

python lander/grid_search.py -c config_sarsa.yaml -g grid_search_sarsa_bins.yaml
python lander/grid_search.py -c config_sarsa.yaml -g grid_search_sarsa_greedy.yaml
python lander/grid_search.py -c config_qlearn.yaml -g grid_search_qlearn_bins.yaml
python lander/grid_search.py -c config_qlearn.yaml -g grid_search_qlearn_greedy.yaml
python lander/grid_search.py -c config_qlearn.yaml -g grid_search_qlearn_lr.yaml
