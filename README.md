# Deep-Reinforcement-Learning

This repository contains several neural network models that implement several Deep Reinforcement Learning algorithms. Reinforcement learning is an area of machine learning concerned with how AI agents ought to take actions in an environment so as to maximize some notion of cumulative reward. In this particular case reinforcement learning algorithms are extended by neural networks. I use OpenAI Gym simulation environments to solve several problems. 

## Deep Q-Learning / Double Q-Learning

Deep Q-Learning is used to teach an AI to solve the cartpole problem


> **Problem Discription**: A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The system is controlled by applying a force of +1 or -1 to the cart. The pendulum starts upright, and the goal is to prevent it from falling over. A reward of +1 is provided for every timestep that the pole remains upright. The episode ends when the pole is more than 15 degrees from vertical, or the cart moves more than 2.4 units from the center.

To run the model execute `src/q learning/run_training.py`. for vanilla Q-Learning implementation and `src/double q learning/run_training.py` if you want to try out the more advanced Double Q-Learning version.


AI agent before training:            |  AI agent after training:
:-------------------------:|:-------------------------:
![alt text](https://github.com/artem-oppermann/Deep-Reinforcement-Learning/blob/master/gif%20samples/cartpole_before.gif)  |  ![alt text](https://github.com/artem-oppermann/Deep-Reinforcement-Learning/blob/master/gif%20samples/cartpole_after4.gif)


#### AI agent before training:
![alt text](https://github.com/artem-oppermann/Deep-Reinforcement-Learning/blob/master/gif%20samples/cartpole_before.gif) 

#### AI agent after training:

![alt text](https://github.com/artem-oppermann/Deep-Reinforcement-Learning/blob/master/gif%20samples/cartpole_after4.gif)


## Policy Gradients

An other approach to solve an environment is using policy gradients. This method is usefull in continues action spaces, where the AI must decide from an infinite number of possible actions. An example for such a problem is the OpenAI's pendulum.

> **Problem Discription**: The inverted pendulum swingup problem is a classic problem in the control literature. In this version of the problem, the pendulum starts in a random position, and the goal is to swing it up so it stays upright. The AI agents becomes an observation state and must decide to apply a force which is continues and is between -2 and +2.

To run the model execute `src/policy gradients/pendulum_pg.py`. 
#### AI agent before training:

![alt text](https://github.com/artem-oppermann/Deep-Reinforcement-Learning/blob/master/gif%20samples/pendulum_before.gif)

#### AI agent after training:

![alt text](https://github.com/artem-oppermann/Deep-Reinforcement-Learning/blob/master/gif%20samples/pendulum_after.gif)


