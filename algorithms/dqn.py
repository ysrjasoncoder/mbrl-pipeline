import random
from datetime import datetime
import os

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast


class DQN:
    def __init__(self, policy_net, target_net, memory, cfg):
        self.memory = memory
        self.policy_net = policy_net
        self.target_net = target_net
        self.cfg = cfg
        self.epsilon = cfg.epsilon_start
        self.sample_count = 0
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr)
        self.scaler = GradScaler()

    @torch.no_grad()
    def choose_action(self, state):
        self.sample_count += 1
        # 线性衰减 epsilon
        self.epsilon = self.cfg.epsilon_end + (self.cfg.epsilon_start - self.cfg.epsilon_end) * \
                       np.exp(-1. * self.sample_count / self.cfg.epsilon_decay)
        if random.random() > self.epsilon:
            s = torch.tensor(state, dtype=torch.float32, device=self.cfg.device).unsqueeze(0)
            return self.policy_net(s).argmax(dim=1).item()
        else:
            return random.randrange(self.cfg.n_actions)

    @torch.no_grad()
    def predict_action(self, state):
        s = torch.tensor(state, dtype=torch.float32, device=self.cfg.device).unsqueeze(0)
        return self.policy_net(s).argmax(dim=1).item()

    def update(self):
        if len(self.memory) < self.cfg.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.cfg.batch_size)
        s_batch = torch.tensor(states, dtype=torch.float32, device=self.cfg.device)
        a_batch = torch.tensor(actions, dtype=torch.long, device=self.cfg.device).unsqueeze(1)
        r_batch = torch.tensor(rewards, dtype=torch.float32, device=self.cfg.device)
        ns_batch = torch.tensor(next_states, dtype=torch.float32, device=self.cfg.device)
        d_batch = torch.tensor(dones, dtype=torch.float32, device=self.cfg.device)

        with autocast():
            q = self.policy_net(s_batch).gather(1, a_batch).squeeze(1)
            q_next = self.target_net(ns_batch).max(1)[0].detach()
            q_target = r_batch + self.cfg.gamma * q_next * (1 - d_batch)
            loss = F.mse_loss(q, q_target)

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def update_from_transition(self, s, a, r, ns, done):
        """ 用外部给的(s,a,r,ns,done)更新 Q 网络 """
        q      = self.policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        q_next = self.target_net(ns).max(1)[0].detach()
        q_target = r + self.cfg.gamma * q_next * (1 - done)
        loss = F.mse_loss(q, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

    def save_model(self, results_dir):
        torch.save(self.policy_net.state_dict(), os.path.join(results_dir, 'model.pth'))

    def end_episode(self, ep):
        # 更新 Target 网络
        if ep % self.cfg.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())