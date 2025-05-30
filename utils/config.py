# utils/config.py

import torch
import yaml

class Config:
    def __init__(self):
        # 默认超参数
        self.env_name        = 'CartPole-v1'
        self.algo_name       = 'dyna'
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
