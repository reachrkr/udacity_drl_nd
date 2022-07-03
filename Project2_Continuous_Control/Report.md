[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: img/result.JPG "Graph"


# Project 2 - Report
## Continuous Control using DDPG agent

## Enviroment Introduction

![Trained Agent][image1]

In this environment, a double-jointed arm can move to target locations. A reward of **`+0.1`** is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of **`33`** variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between **`-1`** and **`1`**.

The task is episodic, and in order to solve the environment, your agent must get an average score of **`+30`** over **`100`** consecutive episodes.


## Solution / Algorithm

The learning algorithm is implemented in the files `Continuous_Control.ipynb` and `ddpg_agent.py`. Once the agent solves the environment with an average score of **`+30`** over **`100`** consecutive episodes, the actor network and critic network are saved in the files `checkpoint_actor.pth` and `checkpoint_critic.pth`, respectively.

The solution is based on the Udacity implementation [DDPG Pendulum](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum).

Based primarily on value-based and policy-based predictions, it consists of two neural networks.In the paper below, the algorithm is described in detail.

Paper: [Lillicrap, Timothy P., et al. "Continuous control with deep reinforcement learning." arXiv preprint arXiv:1509.02971 (2015).](https://arxiv.org/abs/1509.02971)

The agent minimizes the MSE loss computed between expected and actual rewards using [Adam](https://arxiv.org/abs/1412.6980). 

The weights of the agent's networks are softly updated at every training step.

The environment was solved using the amended DDPG Agent. The Actor-Network consists of two fully connected hidden layers `(33, 400)` and `(400, 300)` with ReLU activation. The final linear output layer produces a vector containing the action values for each possible action.

The Critic-Network consists of two fully connected hidden layers `(33, 400)` and `(404, 300)` with Leaky-ReLU activation. The final linear output layer maps (state, action) pairs to their corresponding Q-values.

At every timestep the experiences (S, A, R, S') for each agent are stored in a replay buffer, while both the actor and critic networks are updated **`10`** times after every **`20`** timesteps.   

### Final Hyperparameters

```
BUFFER_SIZE = int(1e6)
BATCH_SIZE = 128
GAMMA = 0.99
TAU = 1e-3
LR_ACTOR = 1e-4
LR_CRITIC = 1e-4
WEIGHT_DECAY = 0
```

### Training Results

Training finished in **`187`** episodes with an average score of **`30.00`** over the last **`100`** consecutive episodes.

## Future Improvements
Our future efforts could incorporate Trust Region Policy Optimization (TRPO), Truncated Natural Policy Gradient (TNPG), or Proximal Policy Optimization (PPO) which should achieve better performance as discussed in this [paper](https://arxiv.org/abs/1604.06778) for control tasks.

Optimize hyperparameters: particularly the actor and critic learning rates. We can build a pipeline that automatically optimizes hyperparameters instead of manually modifying them.

Using prioritized experience replay which can certainly improved the performance in many cases.

Consider limiting the number of timestep updates in order to reduce noise, as well as to make the training quicker.