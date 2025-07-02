# scripts/test.py

#!/usr/bin/env python3
import os
import argparse
import random
import csv
from datetime import datetime


import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from utils.config import *
from models.mlp import MLP
from algorithms.dqn import DQN
from algorithms.ddpg import DDPG
from algorithms.mpc_controller import MPCController
from envs.base_env import make_env

def test_loop(env, agent, cfg, writer, results_dir):
    rewards, steps = [], []
    for ep in range(1, cfg.test_eps + 1):
        state, _ = env.reset() # seed=cfg.seed + ep + 1000
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

def get_latest_timestamped_folder(root_dir: str):
    """
    Find all subfolders named with YYYYMMDD_HHMMSS under root_dir, 
    and return the folder name that comes later in time; 
    if there is no matching folder, return None.
    """
    time_fmt = "%Y%m%d_%H%M%S"
    
    candidates = []
    for name in os.listdir(root_dir):
        full_path = os.path.join(root_dir, name)
        if not os.path.isdir(full_path):
            continue
        try:
            ts = datetime.strptime(name, time_fmt)
            candidates.append((ts, name))
        except ValueError:
            continue

    if not candidates:
        raise ValueError(f"{root_dir} don't has any candiates")
    
    latest_name = max(candidates, key=lambda x: x[0])[1]
    return latest_name

def build_agent(cfg):
    if cfg.agent_name.lower() == 'dqn':
        policy_net = MLP(cfg.n_states, cfg.n_actions, cfg.hidden_dim).to(cfg.device)
        #target_net = MLP(cfg.n_states, cfg.n_actions, cfg.hidden_dim).to(cfg.device)
        #target_net.load_state_dict(policy_net.state_dict())
        agent  = DQN(policy_net, None, None, cfg)
    elif cfg.agent_name.lower() ==  'ddpg':
        agent = DDPG(cfg)
    else:
        raise ValueError(f'Unknown Agent type: {cfg.model_type}')
    return agent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env',   type=str, default='CartPole-v1')
    parser.add_argument('--algo',  type=str, default='dyna')
    parser.add_argument('--model', type=str, default='mlp')
    parser.add_argument('--seed',  type=int, default=42)
    parser.add_argument('--ts',    type=str, default=None,help='The timestamp of the result directory during training, for example 20250530_123456. The latest experiment record is selected by default.')
    args = parser.parse_args()

    # cfg = DQNConfig()
    # cfg.env_name   = args.env
    # cfg.algo_name  = args.algo.lower()
    # cfg.model_type = args.model.lower()
    # cfg.seed       = args.seed

    # # 同样设种子
    # random.seed(cfg.seed)
    # np.random.seed(cfg.seed)
    # torch.manual_seed(cfg.seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(cfg.seed)

    # 构造结果目录
    if args.ts is not None:
        results_dir = os.path.join('results', args.env, args.algo.lower(), args.ts)
    else:
        tmp_dir = os.path.join('results',args.env, args.algo.lower())
        results_dir = os.path.join(tmp_dir, get_latest_timestamped_folder(tmp_dir))

    print(f"Loading the Result in : {results_dir}")

    cfg = Config.load(os.path.join(results_dir, 'config.yaml'))
    cfg.show()

    cfg.device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # TensorBoard（可选）
    tb_writer = SummaryWriter(log_dir=os.path.join(results_dir, 'tensorboard'))

    # 构建 Env + 加载模型
    env = make_env(cfg)
    
    if cfg.algo_name.lower() == 'mpc':
        # TODO 我直接复制过来了
        from algorithms.mpc_controller import build_mpc_controller
        agent = build_mpc_controller(env, None, cfg)
    else:
        agent = build_agent(cfg)
    agent.load_model(results_dir)

    # 运行测试
    test_loop(env, agent, cfg, tb_writer, results_dir)

    tb_writer.close()
    env.close()
    print(f"Testing complete! Results saved to:: {results_dir}")

if __name__ == '__main__':
    main()
