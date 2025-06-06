import gymnasium as gym
import random
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from utils.config import DDPGConfig
from utils.replay_buffer import ReplayBuffer
from algorithms.ddpg import DDPG
from envs.base_env import make_env
# class ReplayBuffer:
#     def __init__(self, cfg):
#         self.buffer = np.empty(cfg.memory_capacity, dtype=object)
#         self.size = 0
#         self.pointer = 0
#         self.capacity = cfg.memory_capacity
#         self.batch_size = cfg.batch_size
#         self.device = cfg.device

#     def push(self, transitions):
#         self.buffer[self.pointer] = transitions
#         self.size = min(self.size + 1, self.capacity)
#         self.pointer = (self.pointer + 1) % self.capacity

#     def clear(self):
#         self.buffer = np.empty(self.capacity, dtype=object)
#         self.size = 0
#         self.pointer = 0

#     def sample(self):
#         batch_size = min(self.batch_size, self.size)
#         indices = np.random.choice(self.size, batch_size, replace=False)
#         samples = map(lambda x: torch.tensor(np.array(x), dtype=torch.float32,
#                                              device=self.device), zip(*self.buffer[indices]))
#         return samples




def env_agent_config(cfg):
    env = gym.make(cfg.env_name).unwrapped
    print(f'观测空间 = {env.observation_space}')
    print(f'动作空间 = {env.action_space}')
    cfg.n_states = int(env.observation_space.shape[0])
    cfg.n_actions = int(env.action_space.shape[0])
    cfg.action_bound = env.action_space.high[0]
    agent = DDPG(cfg)
    return env, agent


def train(env, agent, cfg):
    cfg.show()
    rewards, steps = [], []
    for i in range(cfg.train_eps):
        ep_reward, ep_step = 0.0, 0
        state, _ = env.reset(seed=cfg.seed)
        critic_loss, actor_loss = 0.0, 0.0
        for _ in range(cfg.max_steps):
            ep_step += 1
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.memory.push((state, action, reward, next_state, done))
            state = next_state
            c_loss, a_loss = agent.update()
            critic_loss += c_loss
            actor_loss += a_loss
            ep_reward += reward
            if done:
                break
        rewards.append(ep_reward)
        steps.append(ep_step)
        print(f'回合:{i + 1}/{cfg.train_eps}  奖励:{ep_reward:.0f}  步数:{ep_step:.0f}'
              f'  Critic损失:{critic_loss/ep_step:.4f}  Actor损失:{actor_loss/ep_step:.4f}')
    print('完成训练!')
    env.close()
    return rewards, steps


def test(agent, cfg):
    print('开始测试!')
    rewards, steps = [], []
    env = gym.make(cfg.env_name, render_mode='human')
    for i in range(cfg.test_eps):
        ep_reward, ep_step = 0.0, 0
        state, _ = env.reset(seed=cfg.seed)
        for _ in range(cfg.max_steps):
            ep_step += 1
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            state = next_state
            ep_reward += reward
            if terminated or truncated:
                break
        steps.append(ep_step)
        rewards.append(ep_reward)
        print(f'回合:{i + 1}/{cfg.test_eps}, 奖励:{ep_reward:.3f}')
    print('结束测试!')
    env.close()
    return rewards, steps


if __name__ == '__main__':
    cfg = DDPGConfig()
    cfg.env_name = 'Pendulum-v1'
    env = make_env(cfg)
    agent = DDPG(cfg)
    train_rewards, train_steps = train(env, agent, cfg)
    test_rewards, test_steps = test(agent, cfg)