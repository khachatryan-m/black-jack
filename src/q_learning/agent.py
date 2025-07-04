import time
from collections import defaultdict
import numpy as np


class QLearningAgent:
    def __init__(
            self,
            action_count: int,
            learning_rate: float,
            initial_epsilon: float,
            epsilon_decay: float,
            final_epsilon: float,
            discount_factor: float = 0.95,
            decay='linear',
            init_q_values='random'
    ):
        if init_q_values == 'random':
            self.q_values = defaultdict(lambda: np.random.random(action_count))
        else:
            self.q_values = defaultdict(lambda: np.zeros(action_count))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.decay = decay
        self.episode_time_delta = []
        self.start_time = None

        self.training_error = []

    def get_action(self, obs: tuple[int, int, bool], random_action) -> int:
        if np.random.random() < self.epsilon:
            return random_action

        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
            self,
            obs: tuple[int, int, bool],
            action: int,
            reward: float,
            terminated: bool,
            next_obs: tuple[int, int, bool],
    ):
        if self.start_time is None:
            self.start_time = time.time()
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (
                reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] = (
                self.q_values[obs][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)
        self.episode_time_delta.append(round(time.time() - self.start_time, 2))

    def decay_epsilon(self):
        if self.decay == 'linear':
            self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
        elif self.decay == 'exp':
            self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)
