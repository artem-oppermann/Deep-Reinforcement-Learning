# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 17:55:56 2018

@author: @author: Artem Oppermann
"""

import numpy as np
import tensorflow as tf

class DDQN:
    
    def __init__(self, scope, env, target_network, flags, exp_replay):
        """
        This class build a model for either a target or Q-Network and impliments 
        the methods of Deep Double Q-Learning.
        
        :param scope: A string, tells if this model is a Target-, or Q-Network
        :param env: The openAI Gym instance
        :param target_network: The instance of the Target-Network class.
        :param flags: TensorFlow flags which contain thevalues for hyperparameters
        :param exp_replay: The instance of expereince replay class.
        
        """
            
        self.state_size=len(env.observation_space.sample())
        self.action_size = env.action_space.n
        self.FLAGS=flags
        
        # Building the Target-Network
        if scope=='target':
            
            with tf.variable_scope(scope):
                
                self.x=tf.placeholder(tf.float32, shape=(None,4), name='state')  
                self.q=self.build_target_network()
                self.param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/target_network')

        # Building the Q-Network
        elif scope=='q_network':
            
            with tf.variable_scope(scope):

                self.target_network=target_network
                self.exp_replay=exp_replay

                # State placeholder
                self.x=tf.placeholder(tf.float32, shape=(None,4))
                # TD-Target placeholder
                self.target=tf.placeholder(tf.float32, shape=(None,), name='target')
                # Indices of actions which were selected according to the behaviour policy
                self.action_indices=tf.placeholder(tf.int32, shape=(None,2), name='selected_actions')

                # Calculate Q-values with Q-Network
                self.q=self.build_q_network()
                # Get Parameters of the Q-Network
                self.param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/q_network')
                
                # Select Q-values which correpsond to the actions picked by a behaviour policy
                q_behaviour=tf.gather_nd(self.q, self.action_indices)
                
                with tf.name_scope('q_network_loss'):
                    loss=tf.losses.mean_squared_error(self.target,q_behaviour)
            
                with tf.name_scope('q_network_gradients'):
                    self.gradients=tf.gradients(loss, self.param)
            
                with tf.name_scope('train_q_network'):
                    self.train_opt=tf.train.AdamOptimizer(5e-4).apply_gradients(zip(self.gradients,self.param))
            
                with tf.name_scope('update_target_parameters'):     
                    self.update_opt=[tp.assign(lp) for tp, lp in zip(self.target_network.param,self.param)]
                
        else:
            raise ValueError('No network in scope %s'%scope)
                   
    
    def build_target_network(self):
        '''Build the Target-Network'''
        
        with tf.variable_scope('target_network'):
                 
            W1=tf.get_variable('W1', shape=(4,200), initializer=tf.random_normal_initializer())
            W2=tf.get_variable('W2', shape=(200,200), initializer=tf.random_normal_initializer())
            W3=tf.get_variable('W3', shape=(200,2), initializer=tf.random_normal_initializer())
            
            b1=tf.get_variable('b1', shape=(200), initializer=tf.zeros_initializer())
            b2=tf.get_variable('b2', shape=(200), initializer=tf.zeros_initializer())
            
 
            h1=tf.nn.tanh(tf.matmul(self.x, W1)+b1)
            h2=tf.nn.tanh(tf.matmul(h1, W2)+b2)
            q=tf.matmul(h2, W3)
            
        return q
    
    
    def build_q_network(self):
        '''Build the Q-Network'''
        
        with tf.variable_scope('q_network'):
            W1=tf.get_variable('W1', shape=(4,200), initializer=tf.random_normal_initializer())
            W2=tf.get_variable('W2', shape=(200,200), initializer=tf.random_normal_initializer())
            W3=tf.get_variable('W3', shape=(200,2), initializer=tf.random_normal_initializer())
            
            b1=tf.get_variable('b1', shape=(200), initializer=tf.zeros_initializer())
            b2=tf.get_variable('b2', shape=(200), initializer=tf.zeros_initializer())
            
 
            h1=tf.nn.tanh(tf.matmul(self.x, W1)+b1)
            h2=tf.nn.tanh(tf.matmul(h1, W2)+b2)
            q=tf.matmul(h2, W3)
              
        return q
    

    def train_q_network(self):
        '''Train the Q-Network.'''
        
        experience=self.exp_replay.get_experience()
        
        if len(experience['state']) < self.exp_replay.get_min_experience_count():
            return
        
        #pick random indices from the experience memory
        idx = np.random.choice(len(experience['state']), size=self.FLAGS.batch_size, replace=False)
        
        #pick the according collected experience according to the random indices
        state=np.array([experience['state'][i] for i in idx]).reshape(self.FLAGS.batch_size,self.state_size)
        action=[experience['action'][i] for i in idx]
        reward=[experience['reward'][i] for i in idx]
        next_state=np.array([experience['next_state'][i] for i in idx]).reshape(self.FLAGS.batch_size,self.state_size)
        dones=[experience['done'][i] for i in idx]

        ######Deep Double Q-Learning Implementation######
        
        # Q(s',a)-values according to Target-Network
        q_target=self.session.run(self.target_network.q, feed_dict={self.target_network.x:next_state})
        
        # Q(s',a)-values according to Q-Network
        q_values=self.session.run(self.q, feed_dict={self.x:next_state})
        
        # Select actions according to highest Q-values calculated by the Q-Network
        a=np.argmax(q_values, axis=1)
        
        # Use the Target-Network to evaluate the actions picked in the previous line
        q_next=[np.take(q_target[i],a[i]) for i in range(0,self.FLAGS.batch_size)]

        # Calculate the TD-Target
        targets=[r+self.FLAGS.gamma*q if not done else r for r, q, done in zip(reward,q_next,dones)]
        
        # Get the indices of the taken actions in state s according to the behaviour policy
        indices=[[i,action[i]] for i in range(0, len(action))]
    

        feed_dict={self.x:state, 
                   self.target:targets, 
                   self.action_indices:indices,
                   }
        
        self.session.run(self.train_opt,feed_dict)
    
    def get_action(self, state, eps):
        """Calcualte Q(s,a) by the Q-Network, use the epislon-greedy policy to pick an action..
    
        :param s: current state s, given by the environment
        :param eps: value of probability epsilon
        """
        
        if np.random.random()<eps:
            return np.random.choice(self.action_size)
        else:
            return np.argmax(self.session.run(self.q, feed_dict={self.x: state}))
    
    def set_session(self, session):
        '''Sets the session of the appropriate network. '''
        
        self.session=session
                       
    def update_target_parameter(self):
        '''Set the parameter of the target network to the parameter of the Q-network ''' 
        
        self.session.run(self.update_opt)





