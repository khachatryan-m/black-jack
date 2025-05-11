from helpers import initialize_q, Model, epsilon_greedy_policy, PQ_Model
import math
import heapq

def dyna_Q(env, alpha=0.15, gamma=0.8, planning_steps=100, episodes=10000, k=None):
    Q = {}
    M = Model()
    rewards_over_time = []

    for j in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            M.increment_time()
            initialize_q(Q, state)

            action = epsilon_greedy_policy(Q, state)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward

            initialize_q(Q, next_state)

            greedy_next_action = max([Q.get((next_state, a), 0.0) for a in [0, 1]])
            Q[(state, action)] += alpha * (reward + gamma * greedy_next_action - Q[(state, action)])
            M.add(state, action, next_state, reward)

            for _ in range(planning_steps):
                if not M.model:
                    break

                if k is not None:   # for Dyna-Q+ only
                    (s_i, a_i), (s_next, r_i, tau) = M.sample(include_tau=True)
                    bonus = k * math.sqrt(tau)
                else:
                    (s_i, a_i), (s_next, r_i) = M.sample()
                    bonus = 0

                greedy_next_action = max([Q.get((s_next, a), 0.0) for a in [0, 1]])
                Q[(s_i, a_i)] += alpha * (r_i + bonus + gamma * greedy_next_action - Q[(s_i, a_i)])

            state = next_state
            rewards_over_time.append(reward)

    return Q, rewards_over_time


def PS_DynaQ(env, alpha=0.1, gamma=0.95, planning_steps=20, episodes=1000, theta=0.0001):
    Q = {}
    model = PQ_Model()
    PQ = []
    rewards_over_time = []

    for j in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            initialize_q(Q, state)
            action = epsilon_greedy_policy(Q, state)
            next_state, reward, done, _ = env.step(action)
            initialize_q(Q, next_state)
            episode_reward += reward

            greedy_next_action = max([Q.get((next_state, a), 0.0) for a in [0, 1]])
            delta = reward + gamma * greedy_next_action - Q[(state, action)]

            if abs(delta) > theta:
                heapq.heappush(PQ, (-abs(delta), (state, action)))

            model.add(state, action, next_state, reward)


            for _ in range(planning_steps):
                if not PQ:
                    break

                _, (s_i, a_i) = heapq.heappop(PQ)
                next_state, r = model.get(s_i, a_i)
                initialize_q(Q, next_state)

                greedy_next_action = max([Q.get((next_state, a), 0.0) for a in [0, 1]])
                Q[(s_i, a_i)] += alpha * (r + gamma * greedy_next_action - Q[(s_i, a_i)])

                for pred_s, pred_a in model.get_predecessors(s_i):
                    next_s, pred_r = model.get(pred_s, pred_a)
                    initialize_q(Q, next_s)
                    greedy_pred_next_action = max([Q.get((next_s, a), 0.0) for a  in [0, 1]])
                    pred_delta = pred_r + gamma * greedy_pred_next_action - Q[(pred_s, pred_a)]

                    if abs(pred_delta) > theta:
                        heapq.heappush(PQ, (-abs(pred_delta), (pred_s, pred_a)))

            state = next_state
            rewards_over_time.append(reward)

    return Q, rewards_over_time