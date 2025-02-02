from fly import Fly
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import MultivariateNormal


# define network architecture here
class Net(nn.Module):
    def __init__(self, num_obs, num_act):
        """
            Defines the Neural networks 
        """
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
        """
            Does a forward pass of the Actor Network 
        """
        x = self.shared_net(x)
        mu = self.to_mean(x)
        return mu

    def v(self, x):
        """
            Does a forward pass of the Critic Network 
        """
        x = self.shared_net(x)
        x = self.to_value(x)
        return x


class PPO:
    def __init__(self, args):
        self.args = args

        # initialise parameters
        self.env = Fly(args)
        self.num_acts = self.env.num_act # number of actions
        self.num_obs = self.env.num_obs # number of observations
        self.epoch = 5
        self.lr = 0.001 
        self.gamma = 0.99
        self.lmbda = 0.95
        self.clip = 0.2
        self.mini_batch_size = 40960 #24576 #(4096*6)
        self.chuck_number = 16 # Number of mini_chunk in a rollout I think 
        self.mini_chunk_size = self.mini_batch_size // self.args.num_envs
        print("mini_chunk_size: ", self.mini_chunk_size)
        self.rollout_size = self.mini_chunk_size * self.chuck_number # When does it train 
        print("rollout_size: ", self.rollout_size)
         
        self.num_eval_freq = 100 #Print tout les combien de step 


        
        self.mini_batch_number = 0 # this is an index, we call it mini_batch because it returns all the obs, reward etc of all the envs. Not the same mini_batch as self.mini_batch_size
        
        # initialise the buffers to the good values 
        self.all_obs = torch.zeros((self.rollout_size, self.args.num_envs, self.num_obs), device=self.args.sim_device)
        self.all_acts = torch.zeros((self.rollout_size, self.args.num_envs, self.num_acts), device=self.args.sim_device)
        self.all_next_obs = torch.zeros((self.rollout_size, self.args.num_envs, self.num_obs), device=self.args.sim_device)
        self.all_reward = torch.zeros((self.rollout_size, self.args.num_envs, 1), device=self.args.sim_device)
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

        # How much variance we apply to each action 
        action_var = 0.01 if self.args.testing else 0.2 #was 0.1
        self.action_var = torch.full((self.env.num_act,), action_var).to(args.sim_device) 

        self.optim = torch.optim.Adam(self.net.parameters(), lr=self.lr)

    def make_data(self):
        # compute reward-to-go (target)
        with torch.no_grad():
            target = self.all_reward + self.gamma * self.net.v(self.all_next_obs) * self.all_done
            delta = target - self.net.v(self.all_obs)

        # compute advantage       
        advantage = 0.0
        i = self.rollout_size-1
        for delta_t in reversed(delta):
            advantage = self.gamma * self.lmbda * advantage + delta_t
            self.all_advantage[i] = advantage
            i-=1

        return self.all_obs, self.all_acts, self.all_log_prob, target,  self.all_advantage

    def update(self):
        """
            Update actor and critic network
        """
        obs, action, old_log_prob, target, advantage = self.make_data()
        
        for i in range(self.epoch):
            k = 0
            for j in range(self.mini_chunk_size, self.rollout_size, self.mini_chunk_size):
                # mc stands for mini chunk
                obs_mc, action_mc, old_log_prob_mc, target_mc, advantage_mc = obs[k:j], action[k:j], old_log_prob[k:j], target[k:j], advantage[k:j]
                mu = self.net.pi(obs_mc)
                cov_mat = torch.diag(self.action_var)
                # dist = MultivariateNormal(mu, cov_mat) #THIS IS VERY SLOW 
                scale_tril = torch.cholesky(cov_mat) #But this is fast ! 
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
        """
            This is the main loop, it does a forward pass, gives the action to the environement to apply and then updates its 
            obseravtion buffers. If enough step have passed it trains.
        """
        # collect data
        obs = self.env.obs_buf.clone() # I am not sure you need to clone here or not 
        end = self.env.end #See if we need to stop 

        with torch.no_grad():
            mu = self.net.pi(obs)
            cov_mat = torch.diag(self.action_var)
            scale_tril = torch.cholesky(cov_mat) 
            dist = MultivariateNormal(mu, scale_tril=scale_tril)
            action = dist.sample()
            self.all_log_prob[self.mini_batch_number] = dist.log_prob(action)
            action = action.clip(-1, 1)

        # Steps the environement and apply the actions 
        self.env.step(action)
    
        # Get the different observations from the environement 
        self.all_obs[self.mini_batch_number] = obs
        self.all_acts[self.mini_batch_number] = action
        self.all_next_obs[self.mini_batch_number] = self.env.obs_buf
        self.all_reward[self.mini_batch_number] = self.env.reward_buf.unsqueeze(-1)
        self.all_done = (1 - self.env.reset_buf).unsqueeze(-1)
        
        # Caluculates the mean score between two evaluations 
        self.score += torch.mean(self.all_reward[self.mini_batch_number].float()).item() / self.num_eval_freq 
        
        # If not in testing mode we decrease the action variation 
        if not self.args.testing:
            self.action_var = torch.max(0.01 * torch.ones_like(self.action_var), self.action_var - 0.00001) # was 0.00002

        # training mode
        if self.mini_batch_number+1 == self.rollout_size:
            if not self.args.testing:
                print("Training")
                self.update()
            self.mini_batch_number = 0

            # save sometimes
            # self.optim_step % self.args.save_freq == 0 is not so good because opti step goes up by batch not 1 by 1 
            # so it doesn't really work as intended but still works, someone should fix that. 
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
        
        path = self.args.save_path + endofname + ".pth"

        torch.save(self.net.state_dict(), path)
    
    def generate_video(self):
        self.env.generate_video()

    def exit(self):
        self.env.exit()





      
