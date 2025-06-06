# utils/config.py

import torch
import yaml

class Config:
    def __init__(self):
        
        # 自动选设备
        self.device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def save(self, path):
        with open(path, 'w') as f:
            yaml.dump(vars(self), f)

    def show(self):
        print('-'*30, 'CONFIG', '-'*30)
        for k, v in vars(self).items():
            print(f'{k}: {v}')
        print('-'*80)


# class DDPGConfig(Config):
#     """
#     用于 DDPG 的配置类。
#     包含 Actor-Critic 特有的参数。
#     """
#     def __init__(self):
#         super().__init__()
#         self.agent_type         = 'ddpg'
#         # DDPG 特有字段
#         self.actor_r           = 2e-3
#         self.critic_lr          = 5e-3
#         self.sigma              = 0.01
#         self.tau                = 0.005
#         self.actor_hidden_dim   = 256
#         self.critic_hidden_dim  = 256
#         # 动作边界，由 env_agent_config 填充
#         self.action_bound       = None

class DDPGConfig(Config):
    def __init__(self):
        super().__init__()
        self.env_name = 'Pendulum-v1'
        self.algo_name = 'DYNA'
        self.agent_name = 'DDPG'
        self.train_eps = 150
        self.test_eps = 5
        self.max_steps = 200
        self.batch_size = 128
        self.memory_capacity = 10000
        self.lr_a = 2e-3
        self.lr_c = 5e-3
        self.gamma = 0.99
        self.sigma = 0.01
        self.tau = 0.005
        self.seed = 42
        self.actor_hidden_dim = 256
        self.critic_hidden_dim = 256
        self.n_states = None
        self.n_actions = None
        self.action_bound = None

        # 环境模型超参
        self.model_lr        = 1e-3
        self.planning_steps  = 10
        self.hidden_dim      = 256

class DQNConfig(Config):
    def __init__(self):
        super().__init__()
        # 默认超参数
        self.env_name        = 'CartPole-v1'
        self.algo_name       = 'dyna'
        self.agent_name      = 'DQN'
        self.model_type      = 'mlp'
        self.train_eps       = 100
        self.test_eps        = 5
        self.max_steps       = 100000
        self.epsilon_start   = 0.95
        self.epsilon_end     = 0.01
        self.epsilon_decay   = 800
        self.lr              = 0.001
        self.gamma           = 0.99
        self.seed            = 42
        self.batch_size      = 64
        self.memory_capacity = 100000
        self.hidden_dim      = 256
        self.target_update   = 4
        # 环境模型超参
        self.model_lr        = 1e-3
        self.planning_steps  = 10
