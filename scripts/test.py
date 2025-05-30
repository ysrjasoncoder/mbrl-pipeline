# scripts/test.py

#!/usr/bin/env python3
import os
import argparse
import random
import csv

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from utils.config import Config
from envs.cartpole_env import make_env
from models.mlp import MLP
from algorithms.dqn import DQN

def test_loop(env, agent, cfg, writer, results_dir):
    rewards, steps = [], []
    for ep in range(1, cfg.test_eps + 1):
        state, _ = env.reset(seed=cfg.seed + ep + 1000)
        ep_r, ep_s = 0.0, 0
        while True:
            action = agent.predict_action(state)
            state, reward, term, trunc, _ = env.step(action)
            ep_r += reward
            ep_s += 1
            if term or trunc:
                break
        rewards.append(ep_r)
        steps.append(ep_s)
        print(f'[Test] Ep {ep}/{cfg.test_eps}  Reward: {ep_r:.2f}  Steps: {ep_s}')
        writer.add_scalar('test/reward', ep_r, ep)
        writer.add_scalar('test/steps', ep_s, ep)

    # 保存测试指标
    with open(os.path.join(results_dir, 'test_metrics.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['episode', 'reward', 'steps'])
        for i, (r, s) in enumerate(zip(rewards, steps), 1):
            w.writerow([i, r, s])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env',   type=str, default='CartPole-v1')
    parser.add_argument('--algo',  type=str, default='dyna')
    parser.add_argument('--model', type=str, default='mlp')
    parser.add_argument('--seed',  type=int, default=42)
    parser.add_argument('--ts',    type=str, required=True,
                        help='训练时结果目录的时间戳，例如 20250530_123456')
    args = parser.parse_args()

    cfg = Config()
    cfg.env_name   = args.env
    cfg.algo_name  = args.algo.lower()
    cfg.model_type = args.model.lower()
    cfg.seed       = args.seed

    # 同样设种子
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # 构造结果目录
    results_dir = os.path.join('results', cfg.env_name, cfg.algo_name, args.ts)

    # TensorBoard（可选）
    tb_writer = SummaryWriter(log_dir=os.path.join(results_dir, 'tensorboard'))

    # 构建 Env + 加载模型
    env = make_env(cfg.env_name, cfg.seed, render='human')
    cfg.n_states  = env.observation_space.shape[0]
    cfg.n_actions = env.action_space.n

    policy_net = MLP(cfg.n_states, cfg.n_actions, cfg.hidden_dim).to(cfg.device)
    ckpt_path  = os.path.join(results_dir, 'model.pth')
    policy_net.load_state_dict(torch.load(ckpt_path, map_location=cfg.device))

    agent = DQN(policy_net, policy_net, None, cfg)  # test 时只用 policy_net

    # 运行测试
    test_loop(env, agent, cfg, tb_writer, results_dir)

    tb_writer.close()
    env.close()
    print(f"测试完成，结果保存在: {results_dir}")

if __name__ == '__main__':
    main()
