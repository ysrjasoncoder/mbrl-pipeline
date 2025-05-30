import gym

def make_env(env_name, seed, render=None):
    env = gym.make(env_name, render_mode=render)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env
