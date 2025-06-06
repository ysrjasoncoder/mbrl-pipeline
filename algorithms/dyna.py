import os
import csv

import torch
from torch import nn, optim
from torch.nn import functional as F

def model_update(agent, model, model_opt):
    if len(agent.memory) < agent.cfg.batch_size:
        return 0.0
    # 从 replay buffer 最近采样的 batch 构造张量

    states, actions, rewards2, next_states, dones = agent.memory.sample(agent.cfg.batch_size)
    s_batch = torch.tensor(states, dtype=torch.float32, device=agent.cfg.device)
    a_batch = torch.tensor(actions, dtype=torch.long,  device=agent.cfg.device)
    r_batch = torch.tensor(rewards2, dtype=torch.float32, device=agent.cfg.device)
    ns_batch= torch.tensor(next_states, dtype=torch.float32,device=agent.cfg.device)
    a_oh = None
    if agent.cfg.env_name == 'CartPole-v1':
        # 构造 one-hot 动作
        a_oh = F.one_hot(a_batch,agent.cfg.n_actions).float()
    # 前向模型，预测 Δstate 和 reward
    ns_pred, r_pred = model(s_batch, a_batch if a_oh is None else a_oh)
    # 计算模型误差
    loss_m = F.mse_loss(ns_pred, ns_batch ) + F.mse_loss(r_pred, r_batch)
    model_opt.zero_grad()
    loss_m.backward()
    model_opt.step()
    return loss_m.item()


def dyna_train(env, agent, model, cfg, writer, results_dir):
    cfg.show()
    rewards, steps  = [], []
    if model is not None:
        model_opt = optim.Adam(model.parameters(), lr=cfg.model_lr)
    for ep in range(1, cfg.train_eps + 1):
        state, _ = env.reset(seed=cfg.seed + ep)
        ep_reward, ep_step = 0.0, 0

        for _ in range(cfg.max_steps):
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.memory.push((state, action, reward, next_state, done))
            agent.update()
            if model is not None:
            # —— 2）模型网络更新 ——  <— 就在这里插入！
                loss_m = model_update(agent, model, model_opt)
                if(0 < loss_m < 1e-4):
                    for _ in range(cfg.planning_steps):
                        # 只采样状态和动作
                        s_pl, a_pl = agent.memory.sample_state_action(cfg.batch_size)
                        s_pl = torch.tensor(s_pl, dtype=torch.float32, device=cfg.device)
                        a_pl = torch.tensor(a_pl, dtype=torch.long, device=cfg.device)
                        a_pl_oh = F.one_hot(a_pl, cfg.n_actions).float()

                        # 用模型“想象”下一个状态和奖励
                        ns_pl, r_pl = model(s_pl, a_pl_oh)
                        #ns_pl = s_pl + ds_pl
                        done_pl = torch.zeros_like(r_pl, dtype=torch.float32, device=cfg.device)

                        # 用虚拟 transition 再更新 Q 网络
                        agent.update_from_transition(s_pl, a_pl, r_pl, ns_pl, done_pl)

            state = next_state
            ep_reward += reward
            ep_step += 1
            if done:
                break

        agent.end_episode(ep)

        # 
        rewards.append(ep_reward)
        steps.append(ep_step)
        log_items = [f"Ep {ep:3d}/{cfg.train_eps}",
                     f"Reward: {ep_reward:.2f}",
                     f"Steps: {ep_step}"]
        if model is not None:
            log_items.append(f"Model: {loss_m:.6e}")
        print("[Train]"+"  ".join(log_items))
        
        #print(f'[Train] Ep {ep}/{cfg.train_eps}  Reward: {ep_reward:.2f}  Steps: {ep_step}  Epsilon: {agent.epsilon:.3f} Model Loss: {loss_m:.6f}')
        #print(f'[Train] Ep {ep}/{cfg.train_eps}  Reward: {ep_reward:.2f}  Steps: {ep_step}')

        # TensorBoard 
        writer.add_scalar('train/reward', ep_reward, ep)
        writer.add_scalar('train/steps', ep_step, ep)
        if model is not None:
            writer.add_scalar('model/loss', loss_m, ep)

    # Save Policy Model
    agent.save_model(results_dir)

    # Save Metrics
    with open(os.path.join(results_dir, 'train_metrics.csv'), 'w', newline='') as f_csv:
        writer_csv = csv.writer(f_csv)
        writer_csv.writerow(['episode', 'reward', 'steps'])
        for i, (r, s) in enumerate(zip(rewards, steps), 1):
            writer_csv.writerow([i, r, s])

    return rewards, steps