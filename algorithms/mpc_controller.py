import torch
import os
import numpy as np
# ------------------------------
# 6. MPC Controller
# ------------------------------
class MPCController:
    def __init__(self, model, action_strategy, cost_fn, cost_weights, H, N, device, invalid_fn=None):
        self.model = model.to(device).eval()
        self.action_strategy = action_strategy
        self.cost_fn = cost_fn
        self.cost_weights = cost_weights
        self.H = H
        self.N = N
        self.device = device
        self.invalid_fn = invalid_fn or (lambda s,u: False)

    @torch.no_grad()
    def plan(self, current_state):
        s = torch.tensor(current_state, dtype=torch.float32, device=self.device)
        best_cost = float('inf')
        best_u0 = None
        for _ in range(self.N):
            seq = self.action_strategy.sample_sequence(self.H)
            cost = 0.0
            s_pred = s.clone()
            for u in seq:
                a = self.action_strategy.encode(u, self.device)
                s_pred = self.model(s_pred.unsqueeze(0), a).squeeze(0)
                cost += self.cost_fn(s_pred, u, self.cost_weights) 
                if self.invalid_fn(s_pred, u):
                    cost += 1e3
                    break
            if cost < best_cost:
                best_cost = cost
                best_u0 = seq[0]
        return best_u0
    
    def predict_action(self, s):
        return self.plan(s)

    def save_model(self, results_dir):
        torch.save(self.model.state_dict(), os.path.join(results_dir, 'model.pth'))
    
    def load_model(self, results_dir):
        ckpt_path  = os.path.join(results_dir, 'model.pth')
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))


class MPCController_SingleShooting:
    def __init__(self,
                 model,
                 action_strategy, 
                 cost_fn, 
                 cost_weights, 
                 horizon=20,
                 device='cpu',
                 invalid_fn=None,
                 num_iters=100,
                 lr=0.05,
                 ):
        self.action_strategy = action_strategy
        self.H = horizon
        self.num_iters = num_iters
        self.lr = lr
        self.u_min = action_strategy.low[0]
        self.u_max = action_strategy.high[0]
        self.device = device
        self.model = model
        self.cost_fn = cost_fn
        self.cost_weights = cost_weights

    def _simulate_trajectory(self, obs, u_seq):
        """
        
        """
        cost = torch.tensor(0.0, device=self.device)
        # obs: numpy array [cosθ, sinθ, θ̇]
        state = torch.tensor(obs, dtype=torch.float32, device=self.device)

        for t in range(self.H):
            u = torch.clamp(u_seq[t], self.u_min, self.u_max)

            cos_th, sin_th, thdot = state
            th = torch.atan2(sin_th, cos_th)
            cost = cost + (th**2 + 0.1 * thdot**2 + 0.001 * u**2)

            next_state = self.model(state, u.unsqueeze(0))  
            state = next_state

        return cost

    def plan(self, obs):
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        u_seq = torch.zeros(self.H, dtype=torch.float32, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([u_seq], lr=self.lr)

        # GD
        obs_t = obs  
        for _ in range(self.num_iters):
            optimizer.zero_grad()
            cost = self._simulate_trajectory(obs_t, u_seq)
            cost.backward()
            optimizer.step()
            
            with torch.no_grad():
                u_seq.clamp_(self.u_min, self.u_max)
        #print(cost)
        a0 = u_seq[0].detach().cpu().numpy()

        self.model.train()
        for p in self.model.parameters():
            p.requires_grad = True
        return np.array([a0])
    
    def predict_action(self, s):
        return self.plan(s)

    def save_model(self, results_dir):
        torch.save(self.model.state_dict(), os.path.join(results_dir, 'model.pth'))
    
    def load_model(self, results_dir):
        ckpt_path  = os.path.join(results_dir, 'model.pth')
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))

def build_mpc_controller(env, model, cfg):
    from algorithms.mpc import ENV_CONFIGS, DiscreteStrategy, MLPDynamics
    mpccfg = ENV_CONFIGS[cfg.env_name] 
    args = mpccfg.strategy_args.copy()
    if mpccfg.action_strategy_cls is DiscreteStrategy:
        args['action_dim'] = env.action_space.n
    else:
        args['low']  = env.action_space.low
        args['high'] = env.action_space.high

    strategy = mpccfg.action_strategy_cls(**args)

    if model is None:
        model = MLPDynamics(cfg.n_states, cfg.n_actions)

    if mpccfg.action_strategy_cls is DiscreteStrategy:
        agent = MPCController(model, strategy, mpccfg.cost_fn, mpccfg.cost_weights,
                        mpccfg.horizon, mpccfg.num_samples, cfg.device,
                        invalid_fn=mpccfg.invalid_fn)
    else:
        agent = MPCController_SingleShooting(model, strategy, mpccfg.cost_fn, mpccfg.cost_weights,
                        mpccfg.horizon, cfg.device, invalid_fn=mpccfg.invalid_fn)
    agent.reward_threshold = mpccfg.reward_threshold
    return agent