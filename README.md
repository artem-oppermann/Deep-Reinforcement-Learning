# Deep-Reinforcement-Learning

### This repository contains several deep neural network models that implement different deep reinforcement learning algorithms. 

Reinforcement learning is an area of machine learning concerned with how AI agents ought to take actions in an environment so as to maximize some notion of cumulative reward. In deep reinforcement learning these algorithms are extended by deep neural networks. 

 I use deep reinforcement learning to solve solve several (classical control) problems, taken from the OpenAI Gym simulation environments. 

## 1. Deep Q-Learning / Double Q-Learning

Deep Q-Learning applied on the OpenAI's Gym CartPole Problem.

https://towardsdatascience.com/self-learning-ai-agents-part-ii-deep-q-learning-b5ac60c3f47
https://towardsdatascience.com/deep-double-q-learning-7fca410b193a


> **Problem Discription**: A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The system is controlled by applying a force of +1 or -1 to the cart. The pendulum starts upright, and the goal is to prevent it from falling over. A reward of +1 is provided for every timestep that the pole remains upright. The episode ends when the pole is more than 15 degrees from vertical, or the cart moves more than 2.4 units from the center.

To run the model execute `src/q learning/run_training.py`. for vanilla Q-Learning implementation and `src/double q learning/run_training.py` if you want to try out the more advanced Double Q-Learning version.


### AI agent before and after training with Deep (Double) Q-Learning algorithm


<p float="left">
  <img src="https://github.com/artem-oppermann/Deep-Reinforcement-Learning/blob/master/gif%20samples/cartpole_before.gif" width="430">
  <img src="https://github.com/artem-oppermann/Deep-Reinforcement-Learning/blob/master/gif%20samples/cartpole_after4.gif" width="430">
</p>


## 2. Stochastic Policy Gradients


> **Problem Discription**: An underpowered car must climb a one-dimensional hill to reach a target. Unlike MountainCar v0, the action (engine force applied) is allowed to be a continuous value. The target is on top of a hill on the right-hand side of the car. If the car reaches it or goes beyond, the episode terminates. On the left-hand side, there is another hill. Climbing this hill can be used to gain potential energy and accelerate towards the target. On top of this second hill, the car cannot go further than a position equal to -1, as if there was a wall. Hitting this limit does not generate a penalty (it might in a more challenging version).

To run the model execute `src/policy gradients/stochastic/stochastic_pg.py`.
### AI agent after training with stochastic policy gradient algorithm:

<img src="https://github.com/artem-oppermann/Deep-Reinforcement-Learning/blob/master/gif%20samples/mountain_car_14.gif" width="430">


## 3. Deterministic Policy Gradient


> **Problem Discription**: The inverted pendulum swingup problem is a classic problem in the control literature. In this version of the problem, the pendulum starts in a random position, and the goal is to swing it up so it stays upright. The AI agents becomes an observation state and must decide to apply a force which is continues and is between -2 and +2.

To run the model execute `src/policy gradients/deterministic/pendulum_pg.py`. 


### AI agent before and after training with deterministic policy gradient algorithm:


<p float="left">
  <img src="https://github.com/artem-oppermann/Deep-Reinforcement-Learning/blob/master/gif%20samples/pendulum_before.gif" width="430">
  <img src="https://github.com/artem-oppermann/Deep-Reinforcement-Learning/blob/master/gif%20samples/pendulum_after.gif" width="430">
</p>





