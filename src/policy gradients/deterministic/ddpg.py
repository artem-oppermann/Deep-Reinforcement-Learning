# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 19:12:05 2018

@author: Admin
"""


import numpy as np
import tensorflow as tf
import gym
import imageio
from experience_replay import ExperienceReplay
from noise import OrnsteinUhlenbeckActionNoise

class Actor:
    
    
    def __init__(self, scope, target_network,env, flags):
        
        self.FLAGS=flags
        self.env=env
        self.state_size = len(self.env.observation_space.sample())
        
        if scope=='target':
            
            with tf.variable_scope(scope):
                
                self.x=tf.placeholder(tf.float32, shape=(None,self.state_size), name='state')
                self.action=self.action_estimator(scope='policy_target_network')
                self.param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/policy_target_network')
        else:
            
            with tf.variable_scope(scope):
                
                self.target_network=target_network
            
                self.x=tf.placeholder(tf.float32, shape=(None,self.state_size), name='state')
                self.q_network_gradient=tf.placeholder(tf.float32, shape=(None,1), name='q_network_gradients')
                
                self.action=self.action_estimator(scope='policy_network')
                
                self.param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/policy_network')
                   
                with tf.name_scope('policy_gradient'):
                    self.unnormalized_gradients = tf.gradients(self.action, self.param, -self.q_network_gradient)
                    self.policy_gradient=list(map(lambda x: tf.div(x, self.FLAGS.batch_size), self.unnormalized_gradients))
            
                with tf.name_scope('train_policy_network'):
                    self.train_opt=tf.train.AdamOptimizer(self.FLAGS.learning_rate_Actor).apply_gradients(zip(self.policy_gradient,self.param))    
                
                with tf.name_scope('update_policy_target'):     
                    self.update_opt=[tp.assign(tf.multiply(self.FLAGS.tau,lp)+tf.multiply(1-self.FLAGS.tau,tp)) for tp, lp in zip(self.target_network.param,self.param)]
                          
                with tf.name_scope('initialize_policy_target_network'):
                     self.init_target_op=[tp.assign(lp) for tp, lp in zip(self.target_network.param,self.param)]
                    
    def action_estimator(self, scope):
        
        with tf.variable_scope(scope):
            
            h1 = tf.layers.dense(self.x, 8, tf.nn.relu,use_bias=None,
                                 kernel_initializer=tf.random_normal_initializer(),
                                 bias_initializer=tf.zeros_initializer()
                                 )
            h2 = tf.layers.dense(h1, 8, tf.nn.relu,use_bias=None,
                                 kernel_initializer=tf.random_normal_initializer(),
                                 bias_initializer=tf.zeros_initializer())
            
            h3 = tf.layers.dense(h2, 8, tf.nn.relu,use_bias=None,
                                 kernel_initializer=tf.random_normal_initializer(),
                                 bias_initializer=tf.zeros_initializer())
            
            actions = tf.layers.dense(h3, 1, None, kernel_initializer=tf.random_normal_initializer())  
            
            scalled_actions = self.env.action_space.low + tf.nn.sigmoid(actions)*(self.env.action_space.high - self.env.action_space.low)
            
        return scalled_actions
    


    def set_session(self, session):
        self.session=session
    
    def init_target_network(self):
        self.session.run(self.init_target_op)

    def update_target_parameter(self):
         self.session.run(self.update_opt)
        
    def get_action(self, x):
        return self.session.run(self.action, feed_dict={self.x:x})

    def train(self, state, q_gradient):
        
        feed_dict={self.q_network_gradient: q_gradient,
                   self.x:state
                   }
        self.session.run(self.train_opt,feed_dict)
     
  
class Critic:
    
    def __init__(self, scope,target_network,env, flags):
        
        self.state_size=len(env.observation_space.sample())
        self.FLAGS=flags
        
        if scope=='target':
            
            with tf.variable_scope(scope):
                
                self.gamma=0.99
                self.x=tf.placeholder(tf.float32, shape=(None,self.state_size), name='state')
                self.actions=tf.placeholder(tf.float32, shape=(None,1), name='actions')
                
                self.q=self.action_value_estimator(scope='q_target_network')
                    
                self.param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/q_target_network')

        else:
            
            with tf.variable_scope(scope):
                
                self.target_network=target_network
                self.x=tf.placeholder(tf.float32, shape=(None,self.state_size), name='state')
                self.target=tf.placeholder(tf.float32, shape=(None,1), name='target')
                self.actions=tf.placeholder(tf.float32, shape=(None,1), name='actions')
                
                self.q=self.action_value_estimator(scope='q_network')
                    
                self.param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/q_network')
                
                with tf.name_scope('q_network_loss'):
                    loss=tf.losses.mean_squared_error(self.target,self.q)
            
                with tf.name_scope('q_network_gradient'):
                    self.gradients=tf.gradients(self.q, self.actions)
            
                with tf.name_scope('train_q_network'):
                    self.train_opt=tf.train.AdamOptimizer(self.FLAGS.learning_rate_Critic).minimize(loss)
            
                with tf.name_scope('update_q_target'):     
                    self.update_opt=[tp.assign(tf.multiply(self.FLAGS.tau,lp)+tf.multiply(1-self.FLAGS.tau,tp)) for tp, lp in zip(self.target_network.param,self.param)]
                    
                with tf.name_scope('initialize_q_target_network'):
                     self.init_target_op=[tp.assign(lp) for tp, lp in zip(self.target_network.param,self.param)]
 
    def action_value_estimator(self, scope):    
        
        state_action = tf.concat([self.x, self.actions], axis=1)
        
        with tf.variable_scope(scope):
             
            h1 = tf.layers.dense(state_action, 8, tf.nn.relu,use_bias=None,
                                 kernel_initializer=tf.random_normal_initializer(),
                                 bias_initializer=tf.zeros_initializer()
                                 )
            h2 = tf.layers.dense(h1, 8, tf.nn.relu,use_bias=None,
                                 kernel_initializer=tf.random_normal_initializer(),
                                 bias_initializer=tf.zeros_initializer())
            
            h3 = tf.layers.dense(h2, 8, tf.nn.relu,use_bias=None,
                                 kernel_initializer=tf.random_normal_initializer(),
                                 bias_initializer=tf.zeros_initializer())
            q= tf.layers.dense(h3, 1, None,
                                kernel_initializer=tf.random_normal_initializer())                     
        return q
    
    
    def compute_gradients(self, state, actions):
        
        feed_dict={self.x:state, 
                   self.actions:actions
                   }
        
        q_gradient=self.session.run(self.gradients, feed_dict)
        q_gradient=np.array(q_gradient).reshape(self.FLAGS.batch_size,1)
        
        return q_gradient
    
    
    def calculate_Q(self, state, actions):
        
        feed_dict={self.x: state,
                   self.actions:actions}
        
        q_next=self.session.run(self.q,feed_dict)
        
        return q_next
    
    
    def train(self, state, targets, action):
        
        feed_dict={self.x:state, 
                   self.target:targets, 
                   self.actions:action
                   }
        self.session.run(self.train_opt,feed_dict)
    
    
    def set_session(self, session):
        self.session=session
    
    
    def init_target_network(self):
       self.session.run(self.init_target_op)
             
       
    def update_target_parameter(self):
        self.session.run(self.update_opt)
    

class Model:
    
    def __init__(self, FLAGS):
        
        self.FLAGS=FLAGS
        
        self.env = gym.make('Pendulum-v0')
        self.state_size = len(self.env.observation_space.sample())
        self.num_episodes=1000
        self.batch_size=64
        
        self.exp_replay=ExperienceReplay(50000,1500, FLAGS)
        
        self.action_noise=OrnsteinUhlenbeckActionNoise(self.env,mu= 0.0, sigma=0.2, theta=.15, dt=1e-2, x0=None)
        
        self.actor_target=Actor(scope='target',target_network=None,env=self.env, flags=FLAGS)
        self.actor=Actor(scope='policy',target_network=self.actor_target,env=self.env, flags=FLAGS)
        
        self.critic_target=Critic(scope='target',target_network=None,env=self.env, flags=FLAGS)
        self.critic=Critic(scope='q',target_network=self.critic_target,env=self.env, flags=FLAGS)
        
        init = tf.global_variables_initializer()
        self.session = tf.InteractiveSession()
        self.session.run(init)
        
        self.critic.set_session(self.session)
        self.actor.set_session(self.session)
        self.actor_target.set_session(self.session)
        self.critic_target.set_session(self.session)
        
        self.critic.init_target_network()
        self.actor.init_target_network()
        
    
    def train_networks(self):
        
        if len(self.exp_replay.experience['state']) < self.exp_replay.min_experience:
            return
    
    
        idx = np.random.choice(len(self.exp_replay.experience['state']), size=self.FLAGS.batch_size, replace=False)
        
        state=np.array([self.exp_replay.experience['state'][i] for i in idx]).reshape(self.FLAGS.batch_size,self.state_size)
        action=np.array([self.exp_replay.experience['action'][i] for i in idx]).reshape(self.FLAGS.batch_size,1)
        reward=[self.exp_replay.experience['reward'][i] for i in idx]
        next_state=np.array([self.exp_replay.experience['next_state'][i] for i in idx]).reshape(self.FLAGS.batch_size,self.state_size)
        dones=[self.exp_replay.experience['done'][i] for i in idx]

        #Train critic network
        next_actions=self.actor_target.get_action(next_state)
        q_next=self.critic.target_network.calculate_Q(next_state, next_actions)
        targets=np.array([r+self.gamma*q if not done else r for r, q, done in zip(reward,q_next,dones)])
        self.critic.train(state, targets, action)
        
        #Train actor network
        current_actions=self.actor.get_action(state)        
        q_gradient=self.critic.compute_gradients(state, current_actions)
        self.actor.train(state, q_gradient)
        
        self.actor.update_target_parameter()
        self.critic.update_target_parameter()
        

    def playEpisode(self,episode):
        
        state=self.env.reset()
        state=state.reshape(1,self.state_size)
        done=False
        total_reward=0
   
        while not done:

            action=self.actor.get_action(state)+self.action_noise.get_noise(episode)
            prev_state=state
            state, reward, done, _ = self.env.step(action)
            state=state.reshape(1,self.state_size)
            
            #self.env.render(mode='rgb_array')
            total_reward=total_reward+reward

            self.exp_replay.addExperience(prev_state, action, reward, state, done)
            self.train_networks()
            
        return total_reward
            

    def run_model(self):
        
        totalrewards = np.empty(self.num_episodes+1)
        n_steps=10
        
        for n in range(0, self.num_episodes+1):
            
            total_reward=self.playEpisode(n)
            
            totalrewards[n]=total_reward 
            
            if n>0 and n%n_steps==0:
                print("episodes: %i, avg_reward (last: %i episodes): %.2f" %(n, n_steps, totalrewards[max(0, n-n_steps):(n+1)].mean()))



tf.app.flags.DEFINE_float('learning_rate_Actor', 0.001, 'Learning rate for the policy estimator')

tf.app.flags.DEFINE_float('learning_rate_Critic', 0.001, 'Learning rate for the state-value estimator')

tf.app.flags.DEFINE_float('gamma', 0.99, 'Future discount factor')

tf.app.flags.DEFINE_integer('batch_size', 64, 'Batch size')

tf.app.flags.DEFINE_integer('tau', 1e-2, 'Update rate for the target networks parameter')

FLAGS = tf.app.flags.FLAGS


if __name__ == "__main__":
    pendulum=Model(FLAGS)
    pendulum.run_model()
  
    

        

        
        
        
        
        
        
        
        
        