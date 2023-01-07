from ppo import PPO
import torch
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--sim_device', type=str, default="cuda:0", help='Physics Device in PyTorch-like syntax')
parser.add_argument('--compute_device_id', default=0, type=int)
parser.add_argument('--graphics_device_id', type=int, default=0, help='Graphics Device ID')
parser.add_argument('--num_envs', default=1000, type=int)
parser.add_argument('--headless', default=False)
parser.add_argument('--save', type=bool, default=False)
parser.add_argument('--save_path', type=str, default=None)
parser.add_argument('--save_freq', type=int, default=200)
parser.add_argument('--load', type=bool, default=False) ##Quand ça load ça devrai pas commencer à 0 mais bon osef 
parser.add_argument('--load_path', type=str, default=None)
parser.add_argument('--record', type=bool, default=False) 
parser.add_argument('--record_dir_name', type=str, default=None)
parser.add_argument('--time_steps_per_recorded_frame', type=int, default=2)
parser.add_argument('--testing', type=bool, default=False)

args = parser.parse_args()

torch.manual_seed(0)
random.seed(0)

if args.save_path != None:
    args.save = True

if args.load_path != None:
    args.load = True

if args.record_dir_name != None:
    args.record = True

if args.testing:
    print("## Careful you are in testing mode, no Training will take place ##")

policy = PPO(args)

def main():
    end = False
    while not end:
        end = policy.run()

if __name__ == '__main__':
    main()
    policy.save()
    policy.generate_video()
    policy.exit()  