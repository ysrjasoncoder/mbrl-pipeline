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

import utils.config as config
from envs.base_env import make_env
from utils.replay_buffer import ReplayBuffer
from models.mlp import MLP
from models.dynamics_model import DynamicsModel
from algorithms.dqn import DQN
from algorithms.dyna import dyna_train
from algorithms.ddpg import DDPG
from algorithms.mpc import mpc_train
from algorithms.mpc_controller import build_mpc_controller
from algorithms.mpc import MLPDynamics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env',   type=str, default='CartPole-v1')
    parser.add_argument('--algo',  type=str, default='dyna')
    parser.add_argument('--model', type=str, default='mlp')
    parser.add_argument('--seed',  type=int, default=42)
    args = parser.parse_args()

    # 1. cfg
    if args.algo.lower() == 'mpc':
        cfg = config.MPCConfig() 
    elif(args.env == 'CartPole-v1'):
        cfg = config.DQNConfig()
    elif(args.env == 'Pendulum-v1'):
        cfg = config.DDPGConfig()
    else:
        raise ValueError(f'Unsupported Enviroment Name: {args.env}')

    cfg.env_name   = args.env
    cfg.algo_name  = args.algo.lower()
    cfg.model_type = args.model.lower()
    cfg.seed       = args.seed

    # 2. Set Global seeds 
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    # 3. Result Dir
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join('results', cfg.env_name, cfg.algo_name, ts)
    os.makedirs(results_dir, exist_ok=True)

     # 4. Save Parameter
    cfg.save(os.path.join(results_dir, 'config.yaml'))
    cfg.show()
   
    cfg.device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 5. TensorBoard
    tb_writer = SummaryWriter(log_dir=os.path.join(results_dir, 'tensorboard'))

    # 6. Construct Env/Agent/Model
    env = make_env(cfg)
    
    # if cfg.model_type.lower() == 'mlp':
    #     if cfg.algo_name == 'dyna':
    #         model = DynamicsModel(cfg.n_states, cfg.n_actions, cfg.hidden_dim).to(cfg.device)
    #     elif cfg.algo_name == 'mpc':
    #         model = MLPDynamics(cfg.n_states, cfg.n_actions)
    # else:
    #     model = None

    #TODO: Figure out the reason of this bug: 
    # if I initialize the model before Agent, 
    # the Reinforcement learning will be failed.

    if cfg.model_type.lower() == 'mlp':
        if cfg.algo_name == 'mpc':
            model = MLPDynamics(cfg.n_states, cfg.n_actions)
    else:
        model = None

    if cfg.algo_name.lower() == 'mpc':
        agent = build_mpc_controller(env, model, cfg)
    elif cfg.agent_name.lower() == 'dqn':
        policy_net = MLP(cfg.n_states, cfg.n_actions, cfg.hidden_dim).to(cfg.device)
        target_net = MLP(cfg.n_states, cfg.n_actions, cfg.hidden_dim).to(cfg.device)
        target_net.load_state_dict(policy_net.state_dict())
        memory = ReplayBuffer(cfg.memory_capacity)
        agent  = DQN(policy_net, target_net, memory, cfg)
    elif cfg.agent_name.lower() ==  'ddpg':
        agent = DDPG(cfg)
    else:
        raise ValueError(f'Unknown Agent type: {cfg.model_type}')
    
    if cfg.model_type.lower() == 'mlp':
        if cfg.algo_name == 'dyna':
            model = DynamicsModel(cfg.n_states, cfg.n_actions, cfg.hidden_dim).to(cfg.device)
    else:
        model = None

    if cfg.algo_name.lower() == 'mpc':
        train_fn = mpc_train
    elif cfg.algo_name.lower() == 'dyna':
        train_fn = dyna_train

    # 7. Train
    train_fn(
        env, agent, model, cfg, tb_writer, results_dir
    )

    tb_writer.close()
    env.close()
    print(f"Training complete! Results saved to:: {results_dir}")

if __name__ == '__main__':
    main()
