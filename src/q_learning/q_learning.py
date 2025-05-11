import gymnasium as gym
from gymnasium import Env
from tqdm import tqdm

from agent import QLearningAgent


def train_qlearning_agent(agent: QLearningAgent, gym_env: Env, n_episodes=1000, max_iters=1500):
    env = gym.wrappers.RecordEpisodeStatistics(gym_env, buffer_length=n_episodes)
    for _ in tqdm(range(n_episodes)):
        obs, info = env.reset()
        done = False

        count = 0
        while not done:
            action = agent.get_action(obs, env.action_space.sample())
            next_obs, reward, terminated, truncated, info = env.step(action)

            agent.update(obs, action, reward, terminated, next_obs)

            done = terminated or truncated
            count += 1
            if count > max_iters:
                break
            obs = next_obs

        agent.decay_epsilon()

    return env