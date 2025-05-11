import numpy as np
import random
import gym
import matplotlib.pyplot as plt
import pandas as pd

def epsilon_greedy_policy(Q, state, epsilon=0.1):
    if np.random.rand() < epsilon:
        return np.random.choice([0, 1])
    return np.argmax([Q.get((state, a), 0.0) for a in [0, 1]])


def initialize_q(Q, state):
    for action in [0, 1]:
        if (state, action) not in Q.keys():
            Q[(state, action)] = 0.0


class Model():
    def __init__(self):
      self.model = {}
      self.time = 0
      self.last_visited = {}

    def add(self, state, action, next_state, reward):
      self.model[(state, action)] = (next_state, reward)
      self.last_visited[(state, action)] = self.time

    def sample(self, include_tau=False):
        s_a = random.choice(list(self.model.keys()))
        next_state, reward = self.model[s_a]
        if include_tau:
            tau = self.time - self.last_visited[s_a]
            return s_a, (next_state, reward, tau)
        else:
            return s_a, (next_state, reward)

    def step(self, state, action):
        return self.model.get((state, action), ((0, 0, False), 0))

    def increment_time(self):
      self.time += 1


def evaluate_policy(env: gym.Env, Q, episodes: int=100):
    total_reward = 0
    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            if (state, 0) not in Q or (state, 1) not in Q:
                action = env.action_space.sample()
            else:
                action = np.argmax([Q[(state, a)] for a in [0, 1]])
            state, reward, done, _ = env.step(action)
            print("Current state: player's sum " + str(state[0]) + ", dealer's card: " + str(state[1]) + ". Action taken: " + str(action))
        total_reward += reward
    return total_reward / episodes


def plot_evaluation_results(results):

    df = pd.DataFrame(results, columns=["alpha", "gamma", "episodes", "avg_reward"])

    plt.figure(figsize=(12, 6))
    for (a, g), group in df.groupby(["alpha", "gamma"]):
        plt.plot(group["episodes"], group["avg_reward"], marker='o', label=f"α={a}, γ={g}")

    plt.xlabel("Number of Episodes")
    plt.ylabel("Average Reward")
    plt.title("Dyna-Q Performance over different episodes with different α and γ")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

class PQ_Model():
    def __init__(self):
        self.model = {}
        self.predecessors = {}

    def add(self, s, a, next_s, r):
        self.model[(s, a)] = (next_s, r)
        if next_s not in self.predecessors:
            self.predecessors[next_s] = set()
        self.predecessors[next_s].add((s, a))

    def get(self, s, a):
        return self.model.get((s, a), (None, 0))

    def get_predecessors(self, s):
      return self.predecessors.get(s, set())

    def sample(self):
        return random.choice(list(self.model.keys()))