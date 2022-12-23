from dqn import DQN
from ppo import PPO
from ppo_discrete import PPO_Discrete

import torch
import random
import argparse
import cProfile

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default=None, type=str)
parser.add_argument('--port', default=None, type=int)

parser.add_argument('--sim_device', type=str, default="cuda:0", help='Physics Device in PyTorch-like syntax')
parser.add_argument('--compute_device_id', default=0, type=int)
parser.add_argument('--graphics_device_id', type=int, default=0, help='Graphics Device ID')
parser.add_argument('--num_envs', default=10, type=int)
parser.add_argument('--headless', action='store_true')
parser.add_argument('--method', default='dqn', type=str)
parser.add_argument('--save', type=bool, default=False)
parser.add_argument('--path', type=str, default="saved")
parser.add_argument('--save_freq', type=int, default=20000)
parser.add_argument('--load', type=bool, default=False)
parser.add_argument('--load_path', type=str, default=None)

args = parser.parse_args()
args.headless = False

torch.manual_seed(0)
random.seed(0)

if args.method == 'ppo':
    policy = PPO(args)
elif args.method == 'ppo_d':
    policy = PPO_Discrete(args)
elif args.method == 'dqn':
    policy = DQN(args)

def main():
    end = False
    while not end:
        end = policy.run()

if __name__ == '__main__':
    main()
policy.save()