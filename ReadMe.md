# BlackJack Game Project Report

The project has been done for the course DS 232/CS 380 Reinforcement Learning course at AUA for Spring, 2025.

Contributors: Melanya Khachatryan, Armen Ghazaryan


## Environment

The environment selected for the project is blackjack-v1 provided by the Gymnasium library.  The game rules and environment implementation is available via the following link https://gymnasium.farama.org/environments/toy_text/blackjack/ 

To make use of the environment, simply import gymnasium and create the blackjack environment:
```bash
import gym
env = gym.make("Blackjack-v1")
```

# Implemented algorithms
## Dyna_Q

One of the main algorithms used is Dyna-Q. Here, the agent uses its interactions with the environment to learn a model. It takes actions in the real environment based on its current policy. The policy chosen is ε-greedy with several different values experimented as ε. The direct RL update in the algorithm is done via Q-learning algorithm by selecting the greedy action. 
The planning part of the algorithm consists of search control ( random choices from model buffer) where the agent selects a previously visited state action pair. Then, using the learned model, the agent performs planning to predict future states and rewards. After that, the agent uses the algorithm (Q-learning, in our case) to update the value.

The experiments for Dyna-Q and its extensions have been done by tuning some of the parameters: number of episodes, the discount rate γ, the learning rate α. 

best outcome appears to be when we select the discount rate as 0.9, and the learning rate as 0.2 or 0.3 with the latter slightly overperforming. The mean reward is -0.25 which clearly shows that in most cases the algorithm didn’t succeed in winning the game.

## Dyna-Q+ 
The algorithm logic is almost the same as above, however here the kappa is introduced to give a small bonus to the states that haven’t been explored much. The bonus for visiting a state not visited long before is calculated with the formula κ√τ with κ (kappa) being a small scaling factor (in our case it was set to 0.015).
Here the rewards are slightly better as in the case of previous Dyna-Q with the maximum reward equal to around -0.11 and the mean reward being -0.2.

Code-wise, both Dyna_Q and Dyna_Q+ are implemented as a single function. The default is Dyna-Q. To use the Dyna-Q+, kappa argument should also be specified instead of the default 'None' value. 

## Dyna-Q with prioritized sweeping
Another extension of Dyna-Q was also implemented within the framework of the project. The prioritized sweeping extension of Dyna-Q uses heaps to store action-value pairs and perform updates with the maximum change. In our case, we prioritized the updates of the state-action pairs where the change was greater than 0.002. Compared to Dyna-Q and Dyna-Q+ this algorithm performed slightly worse with the mean reward among all the trials being -0.6 and the maximum average reward achieved being -0.45. 

## Q-learning
As a baseline model-free method, we also implemented Q-Learning on the blackjack-v1 environment. In contrast to the Dyna-Q family, Q-Learning learns action-value estimates directly from real experience without any planning or model updates. Q-Learning uses the maximum action-value in its update, regardless of the action actually taken by the behavioral ε-greedy policy. This “off-policy” nature allows learning the optimal policy even while exploring. Annealing ε balances exploration (random actions early on) with exploitation (greedy actions as learning progresses), helping the agent discover high-value policies without getting trapped in suboptimal behavior too quickly.

