# Deep-Reinforcement-Learning

This repository contains several neural network models that implement several Deep Reinforcement Learning algorithms. Reinforcement learning is an area of machine learning concerned with how AI agents ought to take actions in an environment so as to maximize some notion of cumulative reward. In this particular case reinforcement learning algorithms are extended by neural networks. I use OpenAI Gym simulation environments to solve several problems. 

## Deep Q-Learning

Deep Q-Learning is used to teach an AI to solve the cartpole problem


> **Problem Discription**: A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The system is controlled by applying a force of +1 or -1 to the cart. The pendulum starts upright, and the goal is to prevent it from falling over. A reward of +1 is provided for every timestep that the pole remains upright. The episode ends when the pole is more than 15 degrees from vertical, or the cart moves more than 2.4 units from the center.

To run the model execute `src/q learning/run_training.py`. This is an example of the cartpole problem **before** training:

![alt text](https://github.com/artem-oppermann/Deep-Reinforcement-Learning/blob/master/gif%20samples/cartpole_before.gif)

After 3. min training the AI learned to balance the pole:

![alt text](https://github.com/artem-oppermann/Deep-Reinforcement-Learning/blob/master/gif%20samples/cartpole_after4.gif)

![alt text](https://github.com/artem-oppermann/Deep-Reinforcement-Learning/blob/master/gif%20samples/pendulum_before.gif)


![alt text](https://github.com/artem-oppermann/Deep-Reinforcement-Learning/blob/master/gif%20samples/pendulum_after.gif)
