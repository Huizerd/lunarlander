# LunarLander-v2 for AE4311
Implementation of OpenAI Gym's [LunarLander-v2](https://gym.openai.com/envs/LunarLander-v2/) for the AE4311 Advanced Flight Control course at TU Delft.

## Installation
Tested on Ubuntu 18.04, with Python 3.6.5.
```bash
$ sudo apt install -y python3-numpy python3-dev cmake zlib1g-dev libjpeg-dev xvfb xorg-dev python3-opengl libboost-all-dev libsdl2-dev swig
$ git clone https://github.com/Huizerd/lunarlander.git
$ cd lunarlander
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

## Running the simulator
To run the default configuration (random agent, 10 episodes):
```bash
$ python -m lander
```
Of course, other configurations can be specified. See `config.yaml.default` for the default values.
An example configuration, saved under `config.yaml`, would be:
```yaml
# Environment and agent
ENV_NAME: 'LunarLander-v2'
ENV_SEED: 0
AGENT: 'sarsa'

# Data locations
# NOTE: setting RECORD_DIR to an existing directory will overwrite!
# NOTE: CHECKPOINT_DIR can be anything when CONTINUE is False
RECORD_DIR: 'record/test/'
CHECKPOINT_DIR: 'record/test/'

# Run config
EPISODES: 10000
SAVE_EVERY: 100
STATE_BINS: [10, 10, 4, 4, 6, 4, 2, 2]  # per state dimension
STATE_BOUNDS: [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0],
               [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]  # per state dimension
RENDER: False
CONTINUE: False

# Learning config
# Format: start episode, end episode, start value, end value
# NOTE: linear slope between start and end, then constant
E_GREEDY: [0, 500, 0.0, 0.9]
LEARNING_RATE: [0, -1, 0.01, 0.001]  # -1 indicates last episode
DISCOUNT_RATE: [0, -1, 0.9, 0.9]


```

Which would then be called like this:
```bash
$ python -m lander -c config.yaml
```

## Agents
As of now, the available agents are:
- Random
- Sarsa

## Environment
Additional information about the environment can be found on the environment's [webpage](https://gym.openai.com/envs/LunarLander-v2/), or in the [source code](https://github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py).
### Actions
There are four discrete actions the lander can take:
- `0`: Do nothing
- `1`: Fire left thruster
- `2`: Fire main thruster
- `3`: Fire right thruster

### State
The state vector consists of eight variables (in this order) between -1 and 1:
- Lander position in x
- Lander position in y
- Lander velocity in x
- Lander velocity in y
- Lander angle
- Lander angular velocity
- Contact left landing leg
- Contact right landing leg

To make the learning problem (more) tractable, the state can be discretized into a certain number of bins.