import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from utils.replay_buffer import ReplayBuffer
import os


class Actor(nn.Module):
    def __init__(self, cfg):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(cfg.n_states, cfg.actor_hidden_dim)
        self.fc2 = nn.Linear(cfg.actor_hidden_dim, cfg.actor_hidden_dim)
        self.fc3 = nn.Linear(cfg.actor_hidden_dim, cfg.n_actions)
        self.action_bound = torch.tensor(cfg.action_bound, dtype=torch.float32, device=cfg.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x)) * self.action_bound
        return action

class Critic(nn.Module):
    def __init__(self, cfg):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(cfg.n_states + cfg.n_actions, cfg.critic_hidden_dim)
        self.fc2 = nn.Linear(cfg.critic_hidden_dim, cfg.critic_hidden_dim)
        self.fc3 = nn.Linear(cfg.critic_hidden_dim, 1)


    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value


class DDPG:
    def __init__(self, cfg):
        self.cfg = cfg
        self.memory = ReplayBuffer(cfg.memory_capacity)
        self.actor = torch.jit.script(Actor(cfg).to(cfg.device))
        self.actor_target = torch.jit.script(Actor(cfg).to(cfg.device))
        self.critic = torch.jit.script(Critic(cfg).to(cfg.device))
        self.critic_target = torch.jit.script(Critic(cfg).to(cfg.device))
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=cfg.lr_a)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=cfg.lr_c)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.scaler = GradScaler()

    @torch.no_grad()
    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=self.cfg.device).unsqueeze(0)
        action = self.actor(state).squeeze(0).cpu().numpy()
        action += self.cfg.sigma * np.random.randn(self.cfg.n_actions)
        return action
    
    @torch.no_grad()
    def predict_action(self, state):
        return self.choose_action(state)

    def update(self):
        # if self.memory.size < self.cfg.batch_size:
        #     return 0, 0
        states, actions, rewards, next_states, dones = self.memory.sample2(self.cfg.batch_size)
        actions, rewards, dones = actions.view(-1, 1), rewards.view(-1, 1), dones.view(-1, 1)
        
        with autocast():
            next_q_value = self.critic_target(next_states, self.actor_target(next_states))
            target_q_value = rewards + (1 - dones) * self.cfg.gamma * next_q_value 
            critic_loss = torch.mean(F.mse_loss(self.critic(states, actions), target_q_value))
            
        self.critic_optim.zero_grad()
        self.scaler.scale(critic_loss).backward()
        self.scaler.step(self.critic_optim)

        with autocast():
            actor_loss = -torch.mean(self.critic(states, self.actor(states)))
            
        self.actor_optim.zero_grad()
        self.scaler.scale(actor_loss).backward()
        self.scaler.step(self.actor_optim)

        self.scaler.update()
        self.update_params()

        return actor_loss.item(), critic_loss.item()
    
    def update_from_transition(self, state, action, reward, next_state, done):
        action, reward, done = action.view(-1, 1), reward.view(-1, 1), done.view(-1, 1)
        with autocast():
            next_q_value = self.critic_target(next_state, self.actor_target(next_state))
            target_q_value = reward + (1 - done) * self.cfg.gamma * next_q_value 
            critic_loss = torch.mean(F.mse_loss(self.critic(state, action), target_q_value))
            
        self.critic_optim.zero_grad()
        self.scaler.scale(critic_loss).backward()
        self.scaler.step(self.critic_optim)

        with autocast():
            actor_loss = -torch.mean(self.critic(state, self.actor(state)))
            
        self.actor_optim.zero_grad()
        self.scaler.scale(actor_loss).backward()
        self.scaler.step(self.actor_optim)

        self.scaler.update()
        self.update_params()


    def update_params(self):
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.cfg.tau * param.data +
                                    (1. - self.cfg.tau) * target_param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.cfg.tau * param.data +
                                    (1. - self.cfg.tau) * target_param.data)
    
    def save_model(self, results_dir):
        torch.save(self.actor.state_dict(), os.path.join(results_dir, 'model.pth'))

    def load_model(self, results_dir):
        ckpt_path  = os.path.join(results_dir, 'model.pth')
        self.actor.load_state_dict(torch.load(ckpt_path, map_location=self.cfg.device))
    
    def end_episode(self, ep):
        pass

