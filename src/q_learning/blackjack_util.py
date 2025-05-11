import json
from pathlib import Path
from typing import List
import numpy as np
from tqdm import tqdm

def get_sorted_keys() -> List:
    return json.loads(Path('sorted_keys.json').read_text())


def get_mdp_policy_from_gym_policy(policy_gym):
    policy = [None for _ in range(280)]
    state_idx_mapping = get_sorted_keys()
    for state, action in policy_gym.items():
        state = list(state)
        if state in state_idx_mapping:
            policy[state_idx_mapping.index(state)] = action
    return policy


def get_values_from_qtable(q_table):
    policy = {}
    for state, action_vals in q_table.items():
        policy[state] = np.max(action_vals)
    return policy


def map_to_state_indexes(state_vs_vals):
    policy = [None for _ in range(280)]
    state_idx_mapping = get_sorted_keys()
    for state, value in state_vs_vals.items():
        state = list(state)
        if state in state_idx_mapping:
            policy[state_idx_mapping.index(state)] = value
    return policy


def get_policy_from_qtable(q_table):
    policy = {}
    for state, action_vals in q_table.items():
        policy[state] = int(np.argmax(action_vals))
    return policy


def play_using_policy(env, policy, games=200, max_tries=100, log=True):
    rewards_queue = []
    for _ in tqdm(range(games)):
        obs, info = env.reset()
        done = False

        count = 0
        while not done:
            action = policy[obs]
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            obs = next_obs
            count += 1
            if done:
                rewards_queue.append(reward)
            if count > max_tries:
                rewards_queue.append('T')
                break
    success_pct = (rewards_queue.count(1) / games) * 100
    not_finish_pct = (rewards_queue.count('T') / games) * 100
    if log:
        print(f'Success % => {success_pct} %')
        print(f"Didn't finish % => {not_finish_pct} %")
    return rewards_queue, success_pct, not_finish_pct
