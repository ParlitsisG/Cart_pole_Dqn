import gym
import numpy as np
from ray.rllib.algorithms.dqn import DQNConfig


class CustomEnv(gym.Env):
    def __init__(self, env_config: dict):
        # Construct & Init Environment
        self._env = gym.make('CartPole-v1')

        # Define Action Space: 2 Discrete Actions for cartpole
        self.action_space = gym.spaces.Discrete(2)

        # Define State (Observation) Space: A Continuous State Space represented by a vector of size (4,)
        self.observation_space = gym.spaces.Box(
            low=np.array([-4.8, -np.inf, -0.42, -np.inf]),
            high=np.array([4.8, np.inf, 0.42, np.inf]),
            dtype=np.float32
        )

    # Reset Environment & Init Episode
    def reset(self):
        observation = self._env.reset()
        return observation

    def step(self, action):
        observation, reward, done, info = self._env.step(action)
        return observation, reward, done, info

    def render(self, mode: str or None = None):
        self._env.render()


config = DQNConfig()
config.num_steps_sampled_before_learning_starts = 1000
config.train_batch_size = 64
config.replay_buffer_config.update({
    'capacity': 50000
})
# Pause episode and train
config.batch_mode = 'truncate_episodes'

# Disabling Dueling Feature
config.dueling = False  # later try with True

# Setting Epsilon
config.exploration_config.update({
    "initial_epsilon": 0.5,
    "final_epsilon": 0.01,
    "epsilon_timesteps": 1000,
})

# 1 Step per training
# config.rollout_fragment_length = 1

# Set seed to constant value to reproduce results
config.seed = 0

# Gamma is the discount factor in bellman equation
config.gamma = 0.99

# Set learning rate of neural network
config.lr = 0.0005

# Enable gpu
config.num_gpus = 1


agent = config.framework("tf").environment(env=CustomEnv, env_config={}).build()

def evaluate(agent, eval_env, eval_episodes):
    total_rewards = 0.0

    for _ in range(eval_episodes):
        done = False
        observation = eval_env.reset()

        while not done:
            action = agent.compute_single_action(observation=observation)
            observation, reward, done, _ = eval_env.step(action)
            total_rewards += reward

    return total_rewards/eval_episodes

num_steps = 1000
eval_env = CustomEnv(env_config={})
eval_episodes = 5


for i in range(num_steps):
    agent.train()

    if i % 1 == 0:
        average_rewards = evaluate(agent, eval_env, eval_episodes)
        print('i =', i, ', average rewards =', average_rewards)