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
parser.add_argument('--num_envs', default=1000, type=int)
parser.add_argument('--headless', default=False)
parser.add_argument('--method', default='ppo', type=str)
parser.add_argument('--save', type=bool, default=False)
parser.add_argument('--path', type=str, default="saved")
parser.add_argument('--save_freq', type=int, default=100)
parser.add_argument('--load', type=bool, default=False) ##Quand ça load ça devrai pas commencer à 0 mais bon osef 
parser.add_argument('--load_path', type=str, default=None)
parser.add_argument('--record', type=bool, default=False) #MMm
parser.add_argument('--record_dir_name', type=str, default=None)
parser.add_argument('--time_steps_per_recorded_frame', type=int, default=1000)

args = parser.parse_args()

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