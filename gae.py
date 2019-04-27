import math
import random

import gym
import roboschool
import numpy as np


import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.distributions import Normal

from multiprocessing_env import SubprocVecEnv


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


num_envs = 32
env_name = "RoboschoolInvertedPendulum-v1"

def make_env():
    def _thunk():
        env = gym.make(env_name)
        return env

    return _thunk

envs = [make_env() for i in range(num_envs)]
envs = SubprocVecEnv(envs)

env = gym.make(env_name)
