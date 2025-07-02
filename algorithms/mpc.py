import math
from itertools import cycle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import gymnasium as gym
from dataclasses import dataclass
from algorithms.mpc_controller import MPCController
import os
from utils.mpc_memory import Memory
# ------------------------------
# 1. Data Collection
# ------------------------------
def collect_data(env_name='CartPole-v1', num_rollouts=200, rollout_length=200, seed=0):
    '''
    Deprecated: Moved to member functions of Memory
    '''
    env = gym.make(env_name)
    action_space = env.action_space
    state_dim = env.observation_space.shape[0]
    # Determine action_dim for both discrete and continuous
    if hasattr(action_space, 'n'):
        action_dim = action_space.n
    else:
        action_dim = action_space.shape[0]

    np.random.seed(seed)
    data = {'s': [], 'a': [], 's_next': []}

    for rollout in range(num_rollouts):
        state, _ = env.reset(seed=seed + rollout)
        for _ in range(rollout_length):
            # sample raw action
            if hasattr(action_space, 'n'):
                action = np.random.randint(0, action_dim)
            else:
                action = np.random.uniform(action_space.low, action_space.high)

            next_state, _, terminated, truncated, _ = env.step(action)

            data['s'].append(state.copy())
            # store raw action; encoding done later
            data['a'].append(action) #  if not hasattr(action_space, 'n') else action
            data['s_next'].append(next_state.copy())
            state = next_state
            if terminated or truncated:
                break

    env.close()
    for k in data:
        data[k] = np.array(data[k], dtype=np.float32)
    print(f"[Init] Collected {data['s'].shape[0]} transitions from {env_name}.")
    return data, action_dim

# ------------------------------
# 2. Dynamics Model
# ------------------------------
class MLPDynamics(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256]):
        super().__init__()
        dims = [state_dim + action_dim] + hidden_dims + [state_dim]
        layers = []
        for i in range(len(dims) - 2):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.ReLU()]
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)

# ------------------------------
# 3. Action Strategies
# ------------------------------
class ActionStrategyBase:
    def sample_sequence(self, H):
        raise NotImplementedError
    def encode(self, u, device):
        raise NotImplementedError
    def encode_np(self, u):
        raise NotImplementedError

class DiscreteStrategy(ActionStrategyBase):
    def __init__(self, action_dim):
        self.action_dim = action_dim

    def sample_sequence(self, H):
        return np.random.randint(0, self.action_dim, size=H)

    def encode(self, u, device):
        a = torch.zeros((1, self.action_dim), device=device)
        a[0, int(u)] = 1.0
        return a

    def encode_np(self, u):
        one_hot = np.zeros(self.action_dim, dtype=np.float32)
        one_hot[int(u)] = 1.0
        return one_hot

class ContinuousStrategy(ActionStrategyBase):
    def __init__(self, low, high):
        self.low = low
        self.high = high
        self.action_dim = low.shape[0]

    def sample_sequence(self, H):
        return np.random.uniform(self.low, self.high, size=(H, self.action_dim))

    def encode(self, u, device):
        arr = np.array(u, dtype=np.float32).reshape(1, -1)
        return torch.tensor(arr, dtype=torch.float32, device=device)

    def encode_np(self, u):
        return np.array(u, dtype=np.float32)

# ------------------------------
# 4. Cost Functions
# ------------------------------
def cartpole_cost(s_pred, action, cost_weights):
    x, x_dot, th, th_dot = s_pred.cpu().numpy()
    wx, wth, wdx, wdth = cost_weights
    cost = wx*x**2 + wth*th**2 + wdx*x_dot**2 + wdth*th_dot**2
    return cost


def pendulum_cost(s_pred, action, cost_weights):
    cos_th, sin_th, th_dot = s_pred.cpu().numpy()
    th = math.atan2(sin_th, cos_th)
    w_th, w_thdot, w_a = cost_weights
    cost = w_th*(th - 0)**2 + w_thdot*th_dot**2 + w_a*(np.square(action)).sum()
    return cost

# ------------------------------
# 5. Environment Config
# ------------------------------
@dataclass
class EnvConfig:
    action_strategy_cls: type
    strategy_args: dict
    cost_fn: callable
    cost_weights: tuple
    invalid_fn: callable
    horizon: int
    num_samples: int
    reward_threshold: float

ENV_CONFIGS = {
    'CartPole-v1': EnvConfig(
        action_strategy_cls=DiscreteStrategy,
        strategy_args={'action_dim': None},  # to be filled at runtime
        cost_fn=cartpole_cost,
        cost_weights=(0.1, 5.0, 0.01, 0.01),
        invalid_fn=lambda s,u: abs(s[0])>2.4 or abs(s[2])>0.20943951023931953,  # cartpole_cost handles invalid
        horizon=15,
        num_samples=500,
        reward_threshold=500.0
    ),
    'Pendulum-v1': EnvConfig(
        action_strategy_cls=ContinuousStrategy,
        strategy_args={'low': None, 'high': None},  # to be filled
        cost_fn=pendulum_cost,
        cost_weights=(1.0, 0.1, 0.001),
        invalid_fn=lambda s,u: False,
        horizon=20,
        num_samples=500,
        reward_threshold=-200.0  # e.g. tune if below this
    )
}



# ------------------------------
# 7. Training Dynamics
# ------------------------------
def train_dynamics(model, data_rand, data_rl, epochs=5, batch_size=128, lr=1e-4, rand_ratio=0.5, device='cpu', ep_base = 0, writer=None):
    model.to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    ds_rand = TensorDataset(
        torch.from_numpy(data_rand['s']),
        torch.from_numpy(data_rand['a']),
        torch.from_numpy(data_rand['s_next'])
    )
    ds_rl = None
    if data_rl['s'].shape[0] > 0:
        ds_rl = TensorDataset(
            torch.from_numpy(data_rl['s']),
            torch.from_numpy(data_rl['a']),
            torch.from_numpy(data_rl['s_next'])
        )

    n_rand = int(batch_size * rand_ratio)
    n_rand = min(max(n_rand, 1), len(ds_rand))
    n_rl = batch_size - n_rand
    if ds_rl is None or n_rl <= 0:
        n_rand = batch_size
        n_rl = 0
        ds_rl = None

    loader_rand = DataLoader(ds_rand, batch_size=n_rand, shuffle=True)
    itr_rand = cycle(loader_rand)

    itr_rl = None
    if ds_rl is not None:
        loader_rl = DataLoader(ds_rl, batch_size=n_rl, shuffle=True, drop_last=True)
        itr_rl = cycle(loader_rl)

    total_samples = len(ds_rand) + (len(ds_rl) if ds_rl else 0)
    num_steps = math.ceil(total_samples / batch_size)

    for ep in range(1, epochs + 1):
        total_loss = 0.0
        for _ in range(num_steps):
            s_parts, a_parts, s1_parts = [], [], []
            rs, ra, rs1 = next(itr_rand)
            s_parts.append(rs); a_parts.append(ra); s1_parts.append(rs1)
            if itr_rl is not None:
                rls, rla, rls1 = next(itr_rl)
                s_parts.append(rls); a_parts.append(rla); s1_parts.append(rls1)

            s_batch      = torch.cat(s_parts,   dim=0).to(device)
            a_batch      = torch.cat(a_parts,   dim=0).to(device)
            s_next_batch = torch.cat(s1_parts,  dim=0).to(device)

            pred = model(s_batch, a_batch)
            loss = loss_fn(pred, s_next_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * s_batch.size(0)

        avg_loss = total_loss / (num_steps * batch_size)
        print(f"[Train] E{ep}/{epochs}  Loss: {avg_loss:.6f}")
        writer.add_scalar('model/Avg_loss', avg_loss, ep+ep_base)

    return model

def build_cfg(env_name):
    '''
    Deprecated
    '''
    env = gym.make(env_name)
    cfg = ENV_CONFIGS[env_name] 

    # fill runtime-specific strategy_args 
    args = cfg.strategy_args.copy()
    if cfg.action_strategy_cls is DiscreteStrategy:
        args['action_dim'] = env.action_space.n
    else:
        args['low']  = env.action_space.low
        args['high'] = env.action_space.high

    strategy = cfg.action_strategy_cls(**args)

    theta_thresh = None
    if env_name == 'CartPole-v1':
        theta_thresh = env.unwrapped.theta_threshold_radians
    return env, cfg, strategy, theta_thresh

# ------------------------------
# 8. Unified MPC Run
# ------------------------------
def run_mpc(env, model, memory:Memory,cfg,
            episodes=20, finetune_epochs=3,
            batch_size=128, finetune_lr=1e-4,render=False, device='cpu',writer=None):
    
    # env, cfg, strategy = build_cfg(env_name)


    # mpc = MPCController(model, strategy, cfg.cost_fn, cfg.cost_weights,
    #                     cfg.horizon, cfg.num_samples, device,
    #                     invalid_fn=cfg.invalid_fn) 
    from .mpc_controller import build_mpc_controller
    mpc = build_mpc_controller(env, model, cfg)

    for ep in range(1, episodes+1):
        new_data = {'s': [], 'a': [], 's_next': []}
        state, _ = env.reset(seed=1000 + ep)
        done = False
        total_reward = 0.0
        total_step = 0

        while not done:
            if render:
                env.render()

            action = mpc.plan(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            total_step += 1

            # record for dynamics refitting
            a_np = mpc.action_strategy.encode_np(action)
            new_data['s'].append(state.copy())
            new_data['a'].append(a_np)
            new_data['s_next'].append(next_state.copy())
            state = next_state
            

        writer.add_scalar('train/reward', total_reward, ep)
        writer.add_scalar('train/steps', total_step, ep)

        print(f"[Eval] {cfg.env_name} Episode {ep:2d}  Reward: {total_reward:.1f}")

        if new_data['s']:
            # for k in new_data:
            #     arr = np.array(new_data[k], dtype=np.float32)
            #     data_rl[k] = np.concatenate([data_rl[k], arr], axis=0)
            # print(f"  → Collected {len(new_data['s'])} new transitions, data_rl now {data_rl['s'].shape[0]} samples.")
            memory.add_data(new_data)

            if total_reward < mpc.reward_threshold: 
                # model = train_dynamics(model,
                #                        data_rand, data_rl,
                #                        epochs=finetune_epochs,
                #                        batch_size=batch_size,
                #                        lr=finetune_lr,
                #                        rand_ratio=rand_ratio,
                #                        device=device,
                #                        ep_base=total_train_model_ep,
                #                        writer=writer)
                # total_train_model_ep += finetune_epochs
                memory.train_dynamics(model,
                                       epochs=finetune_epochs,
                                       batch_size=batch_size,
                                       lr=finetune_lr,
                                       device=device)

    env.close()

def mpc_train(env, agent, model, cfg, writer, results_dir):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # choose environment here
    # env_name = cfg.env_name

    # data_rand, action_dim = collect_data(env_name=env_name,
    #                                     num_rollouts=200,
    #                                     rollout_length=200,
    #                                     seed=0)

    # initialize empty RL data
    # state_dim = data_rand['s'].shape[1]
    # data_rl = {
    #     's':      np.zeros((0, state_dim), dtype=np.float32),
    #     'a':      np.zeros((0, action_dim), dtype=np.float32),
    #     's_next': np.zeros((0, state_dim), dtype=np.float32),
    # }


    # if env_name == 'CartPole-v1':
    #     # turn data_rand['a'] from shape (N,) into (N, action_dim)
    #     data_rand['a'] = np.eye(action_dim, dtype=np.float32)[ data_rand['a'].astype(int) ]

    # rand_ratio = 0.7

    memory = Memory(env, writer, rand_ratio=0.7, num_rollouts=200,rollout_length=200,seed=0)
    # model = MLPDynamics(state_dim=cfg.n_states, action_dim=cfg.n_actions)
    # model = train_dynamics(model,
    #                        data_rand, data_rl,
    #                        epochs=100,
    #                        batch_size=128,
    #                        lr=1e-4,
    #                        rand_ratio=rand_ratio,
    #                        device=device,
    #                        ep_base=0,
    #                        writer=writer)
    memory.train_dynamics(model,
                           epochs=100,
                           batch_size=128,
                           lr=1e-4,
                           device=device)

    # torch.save(model.state_dict(), os.path.join(results_dir, 'dynamics_initial.pth'))

    run_mpc(env, model,memory,cfg,
            episodes=20,
            finetune_epochs=10,
            batch_size=128,
            finetune_lr=1e-5,
            render=False,
            device=device,
            writer=writer)

    torch.save(model.state_dict(), os.path.join(results_dir, 'model.pth'))
    
    



# ------------------------------
# 9. Main Process
# ------------------------------
def main():
    '''
    Deprecated
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # choose environment here
    env_name = 'CartPole-v1'  # or 'CartPole-v1'

    data_rand, action_dim = collect_data(env_name=env_name,
                                        num_rollouts=200,
                                        rollout_length=200,
                                        seed=0)


    # initialize empty RL data
    state_dim = data_rand['s'].shape[1]
    data_rl = {
        's':      np.zeros((0, state_dim), dtype=np.float32),
        'a':      np.zeros((0, action_dim), dtype=np.float32),
        's_next': np.zeros((0, state_dim), dtype=np.float32),
    }

    if env_name == 'CartPole-v1':
        # turn data_rand['a'] from shape (N,) into (N, action_dim)
        data_rand['a'] = np.eye(action_dim, dtype=np.float32)[ data_rand['a'].astype(int) ]
        # data_rl starts empty but this keeps it consistent
        #data_rl['a']  = np.zeros((0, action_dim), dtype=np.float32)

    rand_ratio = 0.7

    model = MLPDynamics(state_dim=state_dim, action_dim=action_dim)
    model = train_dynamics(model,
                           data_rand, data_rl,
                           epochs=100,
                           batch_size=128,
                           lr=1e-4,
                           rand_ratio=rand_ratio,
                           device=device)
    torch.save(model.state_dict(), f'{env_name}_dynamics_initial.pth')

    run_mpc(env_name, model, data_rand, data_rl,
            episodes=5,
            finetune_epochs=10,
            batch_size=128,
            finetune_lr=1e-5,
            rand_ratio=rand_ratio,
            render=False,
            device=device)

    torch.save(model.state_dict(), f'{env_name}_dynamics_final.pth')


