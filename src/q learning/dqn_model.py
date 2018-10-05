import numpy as np
import tensorflow as tf


'''Main class for building of the Q-, and Target-Networks'''
class DQN:
    
    def __init__(self, scope, env, target_network, flags, exp_replay):
        
        self.state_size=len(env.observation_space.sample())
        self.action_size = env.action_space.n
        self.FLAGS=flags
        
        if scope=='target':
            
            with tf.variable_scope(scope):
                
                self.x=tf.placeholder(tf.float32, shape=(None,4), name='state')  
                self.q=self.build_target_network()
                self.param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/target_network')

        elif scope=='q_network':
            
            with tf.variable_scope(scope):
                
                self.exp_replay = exp_replay
      
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
                    
        else:
            raise ValueError('No network in scope %s avaiable'%scope)
            
    '''Build neural network for the target.'''    
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
    
    '''Build neural network for the Q-value.''' 
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
    
    '''Train the Q-Network.'''
    def train_q_network(self):
        
        experience=self.exp_replay.get_experience()
        
        if len(experience['state']) <self.exp_replay.get_min_experience_count():
            return
        #pick random indices from the experience memory
        idx = np.random.choice(len(experience['state']), size=self.FLAGS.batch_size, replace=False)
        
        #pick the according collected experience according to the random indices
        state=np.array([experience['state'][i] for i in idx]).reshape(self.FLAGS.batch_size,self.state_size)
        action=[experience['action'][i] for i in idx]
        reward=[experience['reward'][i] for i in idx]
        next_state=np.array([experience['next_state'][i] for i in idx]).reshape(self.FLAGS.batch_size,self.state_size)
        dones=[experience['done'][i] for i in idx]

        # use the 'next state' batch and target_network to predict the Q-value
        q=self.session.run(self.target_network.q, feed_dict={self.target_network.x:next_state})
        q_next=np.max(q,axis=1)

        # calculate the target values
        targets=[r+self.FLAGS.gamma*q if not done else r for r, q, done in zip(reward,q_next,dones)]
        #get the indices of the actions that we picked
        indices=[[i,action[i]] for i in range(0, len(action))]

        feed_dict={self.x:state, 
                   self.target:targets, 
                   self.action_indices:indices
                   }
        
        self.session.run(self.train_opt,feed_dict)
    
    '''Get the action by calculating the q value of an possible action by the Q-Network '''
    def get_action(self, X, eps):
        
        if np.random.random()<eps:
            return np.random.choice(self.action_size)
        else:
            return np.argmax(self.session.run(self.q, feed_dict={self.x:X}))
    
    '''Sets the session of the appropriate network. '''
    def set_session(self, session):
        self.session=session
    
    '''Set the parameter of the target network to the parameter of the Q-network '''                
    def update_target_parameter(self):
        self.session.run(self.update_opt)