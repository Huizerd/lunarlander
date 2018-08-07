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
In addition to this, there are various parser arguments that can be supplied:
- `-a`, `--agent`: choose the agent to use, either `random` or `sarsa`
- `-e`, `--episodes`: the number of episodes to run for
- `-r`, `--render`: whether to render and record the environment on-screen

So, for example:
```bash
$ python -m lander -a sarsa -e 10000 -r True
```

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
- Position in x
- Position in y
- Velocity in x
- Velocity in y
- Lander angle
- Contact left leg
- Contact right leg