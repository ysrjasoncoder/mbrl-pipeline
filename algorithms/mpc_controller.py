import torch
import os
# ------------------------------
# 6. MPC Controller
# ------------------------------
class MPCController:
    def __init__(self, model, action_strategy, cost_fn, cost_weights, H, N, device, invalid_fn=None, theta_thresh=None):
        self.model = model.to(device).eval()
        self.action_strategy = action_strategy
        self.cost_fn = cost_fn
        self.cost_weights = cost_weights
        self.H = H
        self.N = N
        self.device = device
        self.invalid_fn = invalid_fn or (lambda s,u: False)
        self.theta_thresh = theta_thresh

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
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.cfg.device))