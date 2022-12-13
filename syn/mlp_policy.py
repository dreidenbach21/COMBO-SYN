import torch
from torch import nn
from torch import optim
import itertools
from torch import distributions

from utils.latent_model import *
from utils.molecular_transformer import *

class MLP_Policy(nn.Module):
    def __init__(self,  **kwargs):
        super().__init__(**kwargs)
        self.mean_net = mlp(input_size = 32, 
             hidden_size = 64, 
             output_size = 32, 
             activation = nn.Tanh(), 
             output_activation = nn.Identity(), 
             n_layers = 4).cuda()
        self.logstd = nn.Parameter(torch.zeros(32, dtype=torch.float32, device = "cuda"))
        self.learning_rate = 1e-4

        self.optimizer = optim.Adam(
            itertools.chain([self.logstd], self.mean_net.parameters()),
            self.learning_rate
        )
        
        self.latent_space = LatentSpaceModel()
        
    def forward(self, observation):
        batch_mean = self.mean_net(observation)
        scale_tril = torch.diag(torch.exp(self.logstd))
        batch_dim = batch_mean.shape[0]
        batch_scale_tril = scale_tril.repeat(batch_dim, 1, 1)
        action_distribution = distributions.MultivariateNormal(
            batch_mean,
            scale_tril=batch_scale_tril,
        )
        return action_distribution
    
    def encode(self, x):
        return self.latent_space.encode(x)
    
    def decode(self,z):
        return self.latent_space.decode(z)
    
    def latent_sample(self, observations, scale = 1.0):
        z, init_z = self.latent_space.encode(observations, verbose = True)
        noise_dist = self(z)
        noise = noise_dist.rsample()
        log_probs = noise_dist.log_prob(noise)
        latent_actions, kl_term = self.latent_space.learned_sample(init_z, noise, scale)
        
        return latent_actions, log_probs, noise_dist, noise
        
    def latent_delta(self, observations, given_actions, scale = 1.0):
        obs, init_z = self.latent_space.encode(observations, verbose = True)
        
        acs, _ = policy.latent_space.encode(given_actions, verbose = True)
        goal_noise = actions - obs # learn the difference
        noise_dist = self(obs)
        goal_log_probs = noise_dist.log_prob(goal_noise)
        live_noise = noise_dist.rsample()
        live_log_probs = noise_dist.log_prob(live_noise)

        goal_actions, kl_term = self.latent_space.learned_sample(init_z, goal_noise, scale)
        live_actions, kl_term = self.latent_space.learned_sample(init_z, live_noise, scale)
        
        return (goal_actions, live_actions), (goal_log_probs, live_log_probs), (goal_noise, live_noise)
        
    def update(self, log_probs, advantages):
        self.optimizer.zero_grad()
#         print("Advantage", advantages)
#         print("log probs", log_probs)
#         print("Batch_loss", batch_loss.shape, batch_loss)
        loss = -1*(log_probs*advantages).mean()
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()