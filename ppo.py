from env import Cartpole
from fly import Fly

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import MultivariateNormal


# define network architecture here
class Net(nn.Module):
    def __init__(self, num_obs=114, num_act=18):
        super(Net, self).__init__()
        # we use a shared backbone for both actor and critic
        """"
        self.shared_net = nn.Sequential(
            nn.Linear(num_obs, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
        )

        # mean and variance for Actor Network
        self.to_mean = nn.Sequential(
            nn.Linear(256,128), #it was 256 before 
            nn.ELU(),
            nn.Linear(128, num_act),
            nn.ELU()
        )

        # value for Critic Network
        self.to_value = nn.Sequential(
            nn.Linear(256, 128), #it was 256 before 
            nn.ELU(),
            nn.Linear(128, 1)
        )
        """
        self.shared_net = nn.Sequential(
            nn.Linear(num_obs, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU()
        )

        # mean and variance for Actor Network
        self.to_mean = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, num_act),
            nn.Tanh()
        )

        # value for Critic Network
        self.to_value = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
        )

    def pi(self, x):
        x = self.shared_net(x)
        mu = self.to_mean(x)
        return mu

    def v(self, x):
        x = self.shared_net(x)
        x = self.to_value(x)
        return x


class PPO:
    def __init__(self, args):
        self.args = args

        # initialise parameters
        self.env = Fly(args)

        self.epoch = 5
        self.lr = 0.001 
        self.gamma = 0.99
        self.lmbda = 0.95
        self.clip = 0.2
        self.mini_batch_size = 12288  #24576 #(4096*6)
        self.chunk_size = 5 # Aucune idée a quoi set was 32
        self.mini_chunk_size = self.mini_batch_size // self.args.num_envs
        self.rollout_size = self.mini_chunk_size * self.chunk_size
        #self.rollout_size = 128 #Quand est-ce que ça train 
         
        self.num_eval_freq = 100 #Print tout les combien de step 

        self.data = []
        self.score = 0
        self.run_step = 0
        self.optim_step = 0

        self.net = Net(self.env.num_obs, self.env.num_act).to(args.sim_device)
        # Load the weights if specified
        if self.args.load:
            self.net.load_state_dict(torch.load(self.args.load_path))
        self.action_var = torch.full((self.env.num_act,), 0.1).to(args.sim_device)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=self.lr)

    def make_data(self):
        # organise data and make batch
        data = []
        for _ in range(self.chunk_size):
            obs_lst, a_lst, r_lst, next_obs_lst, log_prob_lst, done_lst = [], [], [], [], [], []
            for _ in range(self.mini_chunk_size):
                rollout = self.data.pop()
                obs, action, reward, next_obs, log_prob, done = rollout

                obs_lst.append(obs)
                a_lst.append(action)
                r_lst.append(reward.unsqueeze(-1))
                next_obs_lst.append(next_obs)
                log_prob_lst.append(log_prob)
                done_lst.append(done.unsqueeze(-1))

            obs, action, reward, next_obs, done = \
                torch.stack(obs_lst), torch.stack(a_lst), torch.stack(r_lst), torch.stack(next_obs_lst), torch.stack(done_lst)

            # compute reward-to-go (target)
            with torch.no_grad():
                target = reward + self.gamma * self.net.v(next_obs) * done
                delta = target - self.net.v(obs)

            # compute advantage
            advantage_lst = []
            advantage = 0.0
            for delta_t in reversed(delta):
                advantage = self.gamma * self.lmbda * advantage + delta_t
                advantage_lst.insert(0, advantage)

            advantage = torch.stack(advantage_lst)
            log_prob = torch.stack(log_prob_lst)

            mini_batch = (obs, action, log_prob, target, advantage)
            data.append(mini_batch)
        return data

    def update(self):
        # update actor and critic network
        data = self.make_data()

        for i in range(self.epoch):
            print("Epoch: ", i)
            for mini_batch in data:
                obs, action, old_log_prob, target, advantage = mini_batch

                mu = self.net.pi(obs)
                cov_mat = torch.diag(self.action_var)
                dist = MultivariateNormal(mu, cov_mat)
                log_prob = dist.log_prob(action)

                ratio = torch.exp(log_prob - old_log_prob).unsqueeze(-1)

                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantage

                loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.net.v(obs), target)

                self.optim.zero_grad()
                loss.mean().backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                self.optim.step()

                self.optim_step += 1

    def run(self):
        # collect data
        obs = self.env.obs_buf.clone()
        end = self.env.end

        with torch.no_grad():
            mu = self.net.pi(obs)
            cov_mat = torch.diag(self.action_var)
            dist = MultivariateNormal(mu, cov_mat)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            action = action.clip(-1, 1)

        self.env.step(action)
        next_obs, reward, done = self.env.obs_buf.clone(), self.env.reward_buf.clone(), self.env.reset_buf.clone()

        self.data.append((obs, action, reward, next_obs, log_prob, 1 - done))

        self.score += torch.mean(reward.float()).item() / self.num_eval_freq

        self.action_var = torch.max(0.01 * torch.ones_like(self.action_var), self.action_var - 0.00002)

        # training mode
        if len(self.data) == self.rollout_size:
            print("Training")
            self.update()
            # save sometimes
            if self.args.save and self.optim_step % self.args.save_freq == 0 and self.optim_step != 0:
                print("saving...")
                self.save(str(self.optim_step))
                print("saved!")

        # evaluation mode
        if self.run_step % self.num_eval_freq == 0:
            print('Steps: {:04d} | Opt Step: {:04d} | Reward {:.04f} | Action Var {:.04f}'
                  .format(self.run_step, self.optim_step, self.score, self.action_var[0].item()))
            self.score = 0

        self.run_step += 1
        return end
    
    def save(self, endofname = ""):
        #Saves the weights in a dedicated file 
        if(not self.args.save):
            return 
        
        path = self.args.path + endofname + ".pth"

        torch.save(self.net.state_dict(), path)





      
