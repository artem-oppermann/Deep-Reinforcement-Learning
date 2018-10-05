import numpy as np
import tensorflow as tf
import gym
from JSAnimation.IPython_display import display_animation
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import display
import imageio


BATCH_SIZE=64
GAMMA=0.99
NUM_ITER_UPDATE=25


class DQN:
    
    def __init__(self, scope,env,target_network):
        
        self.state_size=len(env.observation_space.sample())
        self.action_size = env.action_space.n
        
        if scope=='target':
            
            with tf.variable_scope(scope):
                
                self.x=tf.placeholder(tf.float32, shape=(None,4), name='state')  
                self.q=self.build_target_network()
                self.param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/target_network')

        else:
            
            with tf.variable_scope(scope):
                
                self.max_experience=10000
                self.min_experience=100
                self.target_network=target_network
                self.experience={'state':[], 'action':[],'reward':[], 'next_state':[], 'done':[]}

                self.x=tf.placeholder(tf.float32, shape=(None,4))
                self.target=tf.placeholder(tf.float32, shape=(None,), name='target')
                self.action_indices=tf.placeholder(tf.int32, shape=(None,2), name='selected_actions')
                self.q=self.build_q_network()
                self.param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/q_network')
                
                with tf.name_scope('q_network_loss'):
                    loss=tf.losses.mean_squared_error(self.target,tf.gather_nd(self.q, self.action_indices))
            
                with tf.name_scope('q_network_gradients'):
                    self.gradients=tf.gradients(loss, self.param)
            
                with tf.name_scope('train_q_network'):
                    self.train_opt=tf.train.AdamOptimizer(5e-4).apply_gradients(zip(self.gradients,self.param))
            
                with tf.name_scope('update_target_parameter'):     
                    self.update_opt=[tp.assign(lp) for tp, lp in zip(self.target_network.param,self.param)]
                
        
    def build_target_network(self):
        
        with tf.variable_scope('target_network'):
            h1 = tf.layers.dense(self.x, 200, tf.nn.tanh,use_bias=True,
                                 kernel_initializer=tf.random_normal_initializer(),
                                 bias_initializer=tf.zeros_initializer()
                                 )
            h2 = tf.layers.dense(h1, 200, tf.nn.tanh,use_bias=True,
                                 kernel_initializer=tf.random_normal_initializer(),
                                 bias_initializer=tf.zeros_initializer())
            q = tf.layers.dense(h2, 2, None,
                                kernel_initializer=tf.random_normal_initializer()) 
        
        return q
    
    def build_q_network(self):
        
        with tf.variable_scope('q_network'):
            h1 = tf.layers.dense(self.x, 200, tf.nn.tanh,use_bias=True,
                                 kernel_initializer=tf.random_normal_initializer(),
                                 bias_initializer=tf.zeros_initializer())
            h2 = tf.layers.dense(h1, 200, tf.nn.tanh,use_bias=True,
                                 kernel_initializer=tf.random_normal_initializer(),
                                 bias_initializer=tf.zeros_initializer())
            q = tf.layers.dense(h2, 2, None,
                                kernel_initializer=tf.random_normal_initializer()) 
        
        return q
    
    def train_q_network(self):
        
        if len(self.experience['state']) < self.min_experience:
            return
    
        idx = np.random.choice(len(self.experience['state']), size=BATCH_SIZE, replace=False)
        
        state=np.array([self.experience['state'][i] for i in idx]).reshape(BATCH_SIZE,self.state_size)
        action=[self.experience['action'][i] for i in idx]
        reward=[self.experience['reward'][i] for i in idx]
        next_state=np.array([self.experience['next_state'][i] for i in idx]).reshape(BATCH_SIZE,self.state_size)
        dones=[self.experience['done'][i] for i in idx]

        q=self.session.run(self.target_network.q, feed_dict={self.target_network.x:next_state})
        q_next=np.max(q,axis=1)

        targets=[r+GAMMA*q if not done else r for r, q, done in zip(reward,q_next,dones)]
        indices=[[i,action[i]] for i in range(0, len(action))]

        feed_dict={self.x:state, 
                   self.target:targets, 
                   self.action_indices:indices
                   }
        
        self.session.run(self.train_opt,feed_dict)
    
    
    def get_action(self, X, eps):
        
        if np.random.random()<eps:
            return np.random.choice(self.action_size)
        else:
            return np.argmax(self.session.run(self.q, feed_dict={self.x:X}))
    
    
    def addExperience(self, state, action, reward, next_state,done):
        
        if len(self.experience)>self.max_experience:
            self.experience['state'].pop(0)
            self.experience['action'].pop(0)
            self.experience['reward'].pop(0)
            self.experience['next_state'].pop(0)
            self.experience['done'].pop(0)

        self.experience['state'].append(state)
        self.experience['action'].append(action)
        self.experience['reward'].append(reward)
        self.experience['next_state'].append(next_state)
        self.experience['done'].append(done)
    
    def set_session(self, session):
        self.session=session
    
                    
    def update_target_parameter(self):
        self.session.run(self.update_opt)



class CartPole:
    
    def __init__(self):
        
        self.env = gym.make('CartPole-v1')
        self.state_size = len(self.env.observation_space.sample())
        self.num_episodes=1000
        
        target_network=DQN(scope='target',env=self.env,target_network=None)
        self.q_network=DQN(scope='q_network',env=self.env,target_network=target_network)

        init = tf.global_variables_initializer()
        session = tf.InteractiveSession()
        session.run(init)
        
        self.q_network.set_session(session)
        target_network.set_session(session)
        

    def playEpisode(self,eps):
        
        state=self.env.reset()
        state=state.reshape(1,self.state_size)
        
        num_iter=0
        done=False
        total_reward=0

        while not done:
            
            action=self.q_network.get_action(state,eps)
            prev_state=state
            state, reward, done, _ = self.env.step(action)
            state=state.reshape(1,self.state_size)
        
            #self.env.render(mode='rgb_array')
            total_reward=total_reward+reward
            
            if done:
                reward=-100

            self.q_network.addExperience(prev_state, action, reward, state, done)
            self.q_network.train_q_network()
            
            num_iter+=1

            if (num_iter% NUM_ITER_UPDATE) == 0:
                self.q_network.update_target_parameter()
            
        return total_reward
            

    def run(self):
        
        totalrewards = np.empty(self.num_episodes+1)
        n_steps=10
        
        for n in range(0, self.num_episodes+1):
            
            eps = 1.0/np.sqrt(n+1)
            total_reward=self.playEpisode(eps)
            
            totalrewards[n]=total_reward 
            
            if n>0 and n%n_steps==0:
                print("episodes: %i, avg_reward (last: %i episodes): %.2f, eps: %.2f" %(n, n_steps, totalrewards[max(0, n-n_steps):(n+1)].mean(), eps))
 
    
if __name__ == "__main__":
    cartPole=CartPole()
    cartPole.run()
    
