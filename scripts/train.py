# scripts/train.py

#!/usr/bin/env python3
import os
import argparse
import random
import csv
from datetime import datetime

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from utils.config import Config
from envs.cartpole_env import make_env
from utils.replay_buffer import ReplayBuffer
from models.mlp import MLP
from models.dynamics_model import DynamicsModel
from algorithms.dqn import DQN
from algorithms.dyna import dyna_train

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env',   type=str, default='CartPole-v1')
    parser.add_argument('--algo',  type=str, default='dyna')
    parser.add_argument('--model', type=str, default='mlp')
    parser.add_argument('--seed',  type=int, default=42)
    args = parser.parse_args()

    # 1. cfg
    cfg = Config()
    cfg.env_name   = args.env
    cfg.algo_name  = args.algo.lower()
    cfg.model_type = args.model.lower()
    cfg.seed       = args.seed

    # 2. 全局种子
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    # 3. 结果目录
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join('results', cfg.env_name, cfg.algo_name, ts)
    os.makedirs(results_dir, exist_ok=True)

    # 4. 保存超参
    cfg.save(os.path.join(results_dir, 'config.yaml'))
    cfg.show()

    # 5. TensorBoard
    tb_writer = SummaryWriter(log_dir=os.path.join(results_dir, 'tensorboard'))

    # 6. 构建 Env/Agent/Model
    env = make_env(cfg.env_name, cfg.seed)
    cfg.n_states  = env.observation_space.shape[0]
    cfg.n_actions = env.action_space.n

    policy_net = MLP(cfg.n_states, cfg.n_actions, cfg.hidden_dim).to(cfg.device)
    target_net = MLP(cfg.n_states, cfg.n_actions, cfg.hidden_dim).to(cfg.device)
    target_net.load_state_dict(policy_net.state_dict())

    memory = ReplayBuffer(cfg.memory_capacity)
    agent  = DQN(policy_net, target_net, memory, cfg)

    if cfg.model_type.lower() == 'mlp':
        model = DynamicsModel(cfg.n_states, cfg.n_actions, cfg.hidden_dim).to(cfg.device)
    else:
        raise ValueError(f'Unknown model type: {cfg.model_type}')

    # 7. 训练
    train_rewards, train_steps = dyna_train(
        env, agent, model, cfg, tb_writer, results_dir
    )

    # 8. 保存训练指标（重复一次，以便外部脚本快速读取）
    with open(os.path.join(results_dir, 'train_metrics.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'reward', 'steps'])
        for i, (r, s) in enumerate(zip(train_rewards, train_steps), 1):
            writer.writerow([i, r, s])

    tb_writer.close()
    env.close()
    print(f"Training complete! Results saved to:: {results_dir}")

if __name__ == '__main__':
    main()
