import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from itertools import cycle
import math
from torch.utils.data import ConcatDataset, WeightedRandomSampler

class Memory :
    def __init__(self, env, writer = None,
                            rand_ratio = 0.7,
                            num_rollouts=200,
                            rollout_length=200,
                            seed=0):
        self.data_rand, state_dim, action_dim ,self.isDiscreteActionSpace= self._collect_data(env,num_rollouts=num_rollouts,rollout_length=rollout_length,seed=seed)
        self.data_rl = {
            's':      np.zeros((0, state_dim), dtype=np.float32),
            'a':      np.zeros((0, action_dim), dtype=np.float32),
            's_next': np.zeros((0, state_dim), dtype=np.float32),
        }

        if self.isDiscreteActionSpace == True:
            # turn data_rand['a'] from shape (N,) into (N, action_dim)
            self.data_rand['a'] = np.eye(action_dim, dtype=np.float32)[ self.data_rand['a'].astype(int) ]

        self.rand_ratio = rand_ratio
        self.total_ep = 0
        self.writer = writer

    def _collect_data(self, env, num_rollouts=200, rollout_length=200, seed=0):
        action_space = env.action_space
        state_dim = env.observation_space.shape[0]
        # Determine action_dim for both discrete and continuous
        if hasattr(action_space, 'n'):
            action_dim = action_space.n
            isDiscreteActionSpace = True
        else:
            action_dim = action_space.shape[0]
            isDiscreteActionSpace = False

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

        for k in data:
            data[k] = np.array(data[k], dtype=np.float32)
        print(f"[Init] Collected {data['s'].shape[0]} transitions.")
        return data, state_dim, action_dim, isDiscreteActionSpace
    
    def train_dynamics(self, model:nn.Module, epochs:int=5, batch_size:int=128, lr:float=1e-4, device:str='cpu'):
        #TODO : Need to refactor :需要进行解构
        model.to(device)
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        ds_rand = TensorDataset(
            torch.from_numpy(self.data_rand['s']),
            torch.from_numpy(self.data_rand['a']),
            torch.from_numpy(self.data_rand['s_next'])
        )
        ds_rl = None
        if self.data_rl['s'].shape[0] > 0:
            ds_rl = TensorDataset(
                torch.from_numpy(self.data_rl['s']),
                torch.from_numpy(self.data_rl['a']),
                torch.from_numpy(self.data_rl['s_next'])
            )

        n_rand = int(batch_size * self.rand_ratio)
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
            if self.writer is not None:
                self.writer.add_scalar('model/Avg_loss', avg_loss, ep + self.total_ep)
        self.total_ep += epochs
        return model
    
    def add_data(self, new_data):
        for k in new_data:
            arr = np.array(new_data[k], dtype=np.float32)
            self.data_rl[k] = np.concatenate([self.data_rl[k], arr], axis=0)
        print(f"  → Collected {len(new_data['s'])} new transitions, data_rl now {self.data_rl['s'].shape[0]} samples.")

    def get_datasets(self):
        """
        Returns two torch.utils.data.Datasets: a random dataset and an RL dataset (possibly empty)
        """
        ds_rand = TensorDataset(
            torch.from_numpy(self.data_rand['s']),
            torch.from_numpy(self.data_rand['a']),
            torch.from_numpy(self.data_rand['s_next']),
        )
        ds_rl = None
        if self.data_rl['s'].shape[0] > 0:
            ds_rl = TensorDataset(
                torch.from_numpy(self.data_rl['s']),
                torch.from_numpy(self.data_rl['a']),
                torch.from_numpy(self.data_rl['s_next']),
            )
        return ds_rand, ds_rl

    def get_mixed_loader(self, batch_size, rand_ratio=0.7, **loader_kwargs):
        """
        Returns a DataLoader with approximately rand_ratio random data and 1-rand_ratio RL data in the batch.
        If RL data is empty, all data is from random dataset.
        """
        ds_rand, ds_rl = self.get_datasets()
        if ds_rl is None:
            return DataLoader(ds_rand, batch_size=batch_size, shuffle=True, **loader_kwargs)

        # 合并 dataset
        ds_all = ConcatDataset([ds_rand, ds_rl])
        n_rand, n_rl = len(ds_rand), len(ds_rl)

        # 为每个样本设定采样权重
        w_rand = rand_ratio / n_rand
        w_rl   = (1 - rand_ratio) / n_rl
        weights = [w_rand] * n_rand + [w_rl] * n_rl

        sampler = WeightedRandomSampler(weights,
                                        num_samples=batch_size,
                                        replacement=True)
        return DataLoader(ds_all,
                          batch_size=batch_size,
                          sampler=sampler,
                          **loader_kwargs)

