import numpy as np
import gymnasium as gym
env = gym.make('CartPole-v1', render_mode="human")
observation, info = env.reset()

DISCRETE_OS_SIZE = [20, 20, 20, 20]
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))