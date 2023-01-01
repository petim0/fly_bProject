from env import Cartpole
from fly import Fly



import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import MultivariateNormal


# define network architecture here
class Net(nn.Module):
    def __init__(self, num_obs, num_act):
        super(Net, self).__init__()
        # we use a shared backbone for both actor and critic
        
        self.shared_net = nn.Sequential(
            nn.Linear(num_obs, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
        )

        # mean and variance for Actor Network
        self.to_mean = nn.Sequential(
            nn.Linear(128,64), #it was 256 before 
            nn.ELU(),
            nn.Linear(64, num_act),
            nn.ELU()
        )

        # value for Critic Network
        self.to_value = nn.Sequential(
            nn.Linear(128, 64), #it was 256 before 
            nn.ELU(),
            nn.Linear(64, 1)
        )

        """
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
        """

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
        self.num_acts = 18 # number of actions
        self.num_obs = 97+self.num_acts # number of observations
        self.num_rewa = 1 # number of reward
        self.epoch = 5
        self.lr = 0.001 
        self.gamma = 0.99
        self.lmbda = 0.95
        self.clip = 0.2
        self.mini_batch_size = 12288 #24576 #(4096*6)
        self.chuck_number = 16 # Nombre de mini_chunk dans un rollout je crois 
        self.mini_chunk_size = self.mini_batch_size // self.args.num_envs
        print("mini_chunk_size: ", self.mini_chunk_size)
        self.rollout_size = self.mini_chunk_size * self.chuck_number #Quand est-ce que ça train 
        print("rollout_size: ", self.rollout_size)
         
        self.num_eval_freq = 100 #Print tout les combien de step 


        #index, we call it mini_batch because it returns all the obs, reward etc of all the envs. Not the same mini_batch as self.mini_batch_size
        self.mini_batch_number = 0
        self.all_obs = torch.zeros((self.rollout_size, self.args.num_envs, self.num_obs), device=self.args.sim_device)
        self.all_acts = torch.zeros((self.rollout_size, self.args.num_envs, self.num_acts), device=self.args.sim_device)
        self.all_next_obs = torch.zeros((self.rollout_size, self.args.num_envs, self.num_obs), device=self.args.sim_device)
        self.all_reward = torch.zeros((self.rollout_size, self.args.num_envs, self.num_rewa), device=self.args.sim_device)
        self.all_done = torch.zeros((self.rollout_size, self.args.num_envs, 1), device=self.args.sim_device)
        self.all_log_prob = torch.zeros((self.rollout_size, self.args.num_envs), device=self.args.sim_device)
        self.all_advantage = torch.zeros((self.rollout_size, self.args.num_envs, 1), device=self.args.sim_device)

        self.score = 0
        self.run_step = 0
        self.optim_step = 0

        self.net = Net(self.env.num_obs, self.env.num_act).to(args.sim_device)
        # Load the weights if specified
        if self.args.load:
            print("loaded from: ", str(self.args.load_path))
            self.net.load_state_dict(torch.load(self.args.load_path))

        self.action_var = torch.full((self.env.num_act,), 0.1).to(args.sim_device)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=self.lr)

    def make_data(self):
        #obs: 20 (mini_chunk_size) * 10 (num_env) * 114 (self.num_obs)
        #a_lst : 20 (mini_chunk_size) * 10 (num_env) * 18 (num_actions)
        #next_obs: 20 (mini_chunk_size) * 10 (num_env) * 114 (self.num_obs)
        #r_lst: 20 (mini_chunk_size) * 10 (num_env) * 1 (self.num_rewa)
        #done_lst: 20 (mini_chunk_size) * 10 (num_env) * 1 (1 int for done or not done)
        #log_prob = 20 (mini_chunk_size) * 10 (num_env)

        #print("obs_lst: ", len(obs_lst), "*" , len(obs_lst[0]), "*", len(obs_lst[0][0]))
        #print("action", len(a_lst), "*" , len(a_lst[0]), "*", len(a_lst[0][0]))
        #print("obs_lst: ", len(r_lst), "*" , len(r_lst[0]), "*", len(r_lst[0][0]))
        #print("obs_lst: ", len(done_lst), "*" , len(done_lst[0]), "*", len(done_lst[0][0]))
        #print("self.all_log_prob", len(self.all_log_prob), "*" , len(self.all_log_prob[0]))

        # compute reward-to-go (target)
        with torch.no_grad():
            target = self.all_reward + self.gamma * self.net.v(self.all_next_obs) * self.all_done
            delta = target - self.net.v(self.all_obs)

        #print("delta: ", len(delta), "*" , len(delta[0]), "*", len(delta[0][0]))
        # compute advantage
       
        advantage = 0.0
        i = self.rollout_size-1
        for delta_t in reversed(delta):
            advantage = self.gamma * self.lmbda * advantage + delta_t
            self.all_advantage[i] = advantage
            i-=1

        return self.all_obs, self.all_acts, self.all_log_prob, target,  self.all_advantage

    def update(self):
        # update actor and critic network
        obs, action, old_log_prob, target, advantage = self.make_data()
        
        for i in range(self.epoch):
            #print("Epoch: ", i)
            k = 0
            for j in range(self.mini_chunk_size, self.rollout_size, self.mini_chunk_size):
                #mc stands for mini chunk
                obs_mc, action_mc, old_log_prob_mc, target_mc, advantage_mc = obs[k:j], action[k:j], old_log_prob[k:j], target[k:j], advantage[k:j]
                mu = self.net.pi(obs_mc)
                cov_mat = torch.diag(self.action_var)
                #dist = MultivariateNormal(mu, cov_mat) #THIS IS SLOW AS FUCK
                scale_tril = torch.cholesky(cov_mat) #But this is speeeed 
                dist = MultivariateNormal(mu, scale_tril=scale_tril)
                log_prob = dist.log_prob(action_mc)

                ratio = torch.exp(log_prob - old_log_prob_mc).unsqueeze(-1)
                surr1 = ratio * advantage_mc
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantage_mc
                loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.net.v(obs_mc), target_mc)

                self.optim.zero_grad()
                loss.mean().backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                self.optim.step()

                self.optim_step += 1
                k = j

    def run(self):
        # collect data
        obs = self.env.obs_buf.clone() #NEED TO CLONE HERE ?? TODO 
        end = self.env.end #See if we need to stop 

        with torch.no_grad():
            mu = self.net.pi(obs)
            cov_mat = torch.diag(self.action_var)
            dist = MultivariateNormal(mu, cov_mat)
            action = dist.sample()
            self.all_log_prob[self.mini_batch_number] = dist.log_prob(action) #This is a tensor 
            action = action.clip(-1, 1) #MMMMM what is this doing ? 

        self.env.step(action)
    
        
        self.all_obs[self.mini_batch_number] = obs
        self.all_acts[self.mini_batch_number] = action
        self.all_next_obs[self.mini_batch_number] = self.env.obs_buf
        self.all_reward[self.mini_batch_number] = self.env.reward_buf.unsqueeze(-1)
        self.all_done = (1 - self.env.reset_buf).unsqueeze(-1)
        

        self.score += torch.mean(self.all_reward[self.mini_batch_number].float()).item() / self.num_eval_freq #IS THIS A TENSOR ?? 

        self.action_var = torch.max(0.01 * torch.ones_like(self.action_var), self.action_var - 0.00002)

        # training mode
        if self.mini_batch_number+1 == self.rollout_size:
            print("Training")
            self.update()
            self.mini_batch_number = 0
            # save sometimes
            if self.args.save and self.optim_step % self.args.save_freq == 0 and self.optim_step != 0:
                print("saving...")
                self.save(str(self.optim_step))
                print("saved!")
        else:
            self.mini_batch_number += 1
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





      
