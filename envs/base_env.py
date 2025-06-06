import gym

def make_env(cfg, render=None):
    env = gym.make(cfg.env_name, render_mode=render)
    env.reset(seed=cfg.seed)
    env.action_space.seed(cfg.seed)
    env.observation_space.seed(cfg.seed)
    if(cfg.env_name == 'CartPole-v1'):
        cfg.n_states  = env.observation_space.shape[0]
        cfg.n_actions = env.action_space.n
    elif(cfg.env_name == 'Pendulum-v1'):
        cfg.n_states = int(env.observation_space.shape[0])
        cfg.n_actions = int(env.action_space.shape[0])
        cfg.action_bound = env.action_space.high[0]
    else:
        raise ValueError(f'Unknown Enviroment Name: {cfg.env_name}')
    return env