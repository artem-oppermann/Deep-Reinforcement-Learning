# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 17:55:56 2018

@author: @author: Artem Oppermann
"""

import numpy as np
import tensorflow as tf
import gym
from ddqn_model import DDQN
from exp_replay import ExperienceReplay


class CartPole:
    """
    This class build the openAI Gym Environments and runs episodes of the environment.
    
    :param FLAGS: TensorFlow flags which contain the values for hyperparameters

    """
    
    def __init__(self, FLAGS):
        
        self.FLAGS=FLAGS
        self.env = gym.make('CartPole-v1')
        self.state_size = len(self.env.observation_space.sample())
        self.num_episodes=1000
        
        # Build the experience replay class
        self.exp_replay = ExperienceReplay()
        
        # Build the Target-Network
        target_network=DDQN(scope='target', env=self.env,target_network=None, flags=FLAGS, exp_replay=None)
        # Build the Q-Network
        self.q_network=DDQN(scope='q_network',env=self.env,target_network=target_network, flags=FLAGS, exp_replay=self.exp_replay)

        init = tf.global_variables_initializer()
        session = tf.InteractiveSession()
        session.run(init)
        
        self.q_network.set_session(session)
        target_network.set_session(session)
        
          
    def playEpisode(self,eps):
        """Play one single episode of the environmet. 
        :param eps: value of probability epsilon
        
        """
        # Init and reshape first state
        state=self.env.reset()
        state=state.reshape(1,self.state_size)
        
        num_iter=0
        done=False
        total_reward=0
        
        while not done:
            
            # Get an action
            action=self.q_network.get_action(state,eps)
            
            prev_state=state
            # Receive the next sate, reward and done
            state, reward, done, _ = self.env.step(action)
            state=state.reshape(1,self.state_size)
        
            #self.env.render(mode='rgb_array')
            total_reward=total_reward+reward
            
            if done:
                reward=-100
            # Add <s,a,r,s',done> tuple to the experience replay memory
            self.exp_replay.addExperience(prev_state, action, reward, state, done)
            self.q_network.train_q_network()
            
            num_iter+=1

            if (num_iter% self.FLAGS.num_iter_update) == 0:
                self.q_network.update_target_parameter()
            
        return total_reward
            

    def run(self):
        """ Main loop for the running of the episodes. """
        
        totalrewards = np.empty(self.num_episodes+1)
        n_steps=10
        
        for n in range(0, self.num_episodes+1):
            
            eps = 1.0/np.sqrt(n+1)
            total_reward=self.playEpisode(eps)
            
            totalrewards[n]=total_reward 
            
            if n>0 and n%n_steps==0:
                print("episodes: %i, avg_reward (last: %i episodes): %.2f, eps: %.2f" %(n, n_steps, totalrewards[max(0, n-n_steps):(n+1)].mean(), eps))
 
    
