# LunarLander-v2 for AE4311
Implementation of OpenAI Gym's LunarLander-v2 for the AE4311 Advanced Flight Control course at TU Delft.

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
```bash
$ python -m lander
```