import argparse
import numpy as np
import os
import time
import pickle
import sys

# Import PyTorch Stuff
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F

# Import our algorithm
from algorithm import ATOC_COMA_trainer


def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple_spread", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=6000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")  # maddpg: global q function, ddpg: local q function
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    parser.add_argument("--tau", type=float, default=0.001, metavar='G',
                        help='discount factor for model (default: 0.001)')
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default='test', help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="/home/ubuntu/maddpg/saved_policy", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=True)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="/home/ubuntu/maddpg/generated_plots", help="directory where plot data is saved")
    return parser.parse_args()


def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world. This will make the world according to the scenario see "simple_spread.py" > make_world
    world = scenario.make_world()
    # create multiagent environment. Now all the functions we need are in the env
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)  # reset, reward, obs are callbacks
    return env


def train(arglist):
    # Create environment. This takess the scenario and creates a world, and creates and environment with all the
    # functions required. This is similar to the AI Gym environment.
    env = make_env(arglist.scenario, arglist, arglist.benchmark)
    trainer = ATOC_COMA_trainer(arglist.gamma, arglist.tau, arglist.num_units, env.observation_space[0], env.action_space[0])


if __name__=="__main__":
    arglist = parse_args()
    train(arglist)