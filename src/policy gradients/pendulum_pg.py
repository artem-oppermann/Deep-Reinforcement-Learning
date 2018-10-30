import numpy as np
import tensorflow as tf
import gym
import imageio

BATCH_SIZE=64
LEARNING_RATE_C=1e-3
LEARNING_RATE_A=1e-3


class ExperienceReplay:
    
    def __init__(self, max_experience,min_experience):
        
        self.max_experience=max_experience
        self.min_experience=min_experience
        
        self.experience={'state':[], 'action':[],'reward':[], 'next_state':[], 'done':[]}
        
    
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
    

    def get_sample(self):
        
        idx = np.random.choice(len(self.experience['state']), size=BATCH_SIZE, replace=False)
        
        state=np.array([self.experience['state'][i] for i in idx]).reshape(BATCH_SIZE,self.state_size)
        action=[self.experience['action'][i] for i in idx]
        reward=[self.experience['reward'][i] for i in idx]
        next_state=np.array([self.experience['next_state'][i] for i in idx]).reshape(BATCH_SIZE,self.state_size)
        dones=[self.experience['done'][i] for i in idx]
        
        return state, action, reward, next_state, dones


class PolicyNetwork:
    
    
    def __init__(self, scope, target_network,env):
        
        self.env=env
        self.state_size = len(self.env.observation_space.sample())
        self.tau=1e-2
        
        if scope=='target':
            
            with tf.variable_scope(scope):
                
                self.x=tf.placeholder(tf.float32, shape=(None,self.state_size), name='state')
                self.action=self.build_target_network()
                self.param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/policy_target_network')
        else:
            
            with tf.variable_scope(scope):
                
                self.target_network=target_network
            
                self.x=tf.placeholder(tf.float32, shape=(None,self.state_size), name='state')
                self.q_network_gradient=tf.placeholder(tf.float32, shape=(None,1), name='q_network_gradients')
                self.action=self.build_policy_network()
                self.param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/policy_network')
                   
                with tf.name_scope('policy_gradient'):
                    self.unnormalized_gradients = tf.gradients(self.action, self.param, -self.q_network_gradient)
                    self.policy_gradient=list(map(lambda x: tf.div(x, BATCH_SIZE), self.unnormalized_gradients))
            
                with tf.name_scope('train_policy_network'):
                    self.train_opt=tf.train.AdamOptimizer(LEARNING_RATE_A).apply_gradients(zip(self.policy_gradient,self.param))    
                
                with tf.name_scope('update_policy_target'):     
                    self.update_opt=[tp.assign(tf.multiply(self.tau,lp)+tf.multiply(1-self.tau,tp)) for tp, lp in zip(self.target_network.param,self.param)]
                          
                with tf.name_scope('initialize_policy_target_network'):
                     self.init_target_op=[tp.assign(lp) for tp, lp in zip(self.target_network.param,self.param)]
                    

    def init_target_network(self):
        self.session.run(self.init_target_op)

    def update_target_parameter(self):
         self.session.run(self.update_opt)
        
    def get_action(self, x):
        return self.session.run(self.action, feed_dict={self.x:x})
        
    def build_target_network(self):
        
        with tf.variable_scope('policy_target_network'):
            
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
    
    def build_policy_network(self):
        
        with tf.variable_scope('policy_network'):
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
            
            actions = tf.layers.dense(h3, 1, None,
                                kernel_initializer=tf.random_normal_initializer())  
            scalled_actions = self.env.action_space.low + tf.nn.sigmoid(actions)*(self.env.action_space.high - self.env.action_space.low)
    
        return scalled_actions

    def set_session(self, session):
        self.session=session
        
  
class QNetwork:
    
    def __init__(self, scope,target_network,env):
        
        self.state_size=len(env.observation_space.sample())
        self.tau=1e-2
        
        if scope=='target':
            
            with tf.variable_scope(scope):
                
                self.gamma=0.99
                self.x=tf.placeholder(tf.float32, shape=(None,self.state_size), name='state')
                self.actions=tf.placeholder(tf.float32, shape=(None,1), name='actions')
                self.q=self.build_target_network()
                self.param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/q_target_network')

        else:
            
            with tf.variable_scope(scope):
                
                self.target_network=target_network
                self.x=tf.placeholder(tf.float32, shape=(None,self.state_size), name='state')
                self.target=tf.placeholder(tf.float32, shape=(None,1), name='target')
                self.actions=tf.placeholder(tf.float32, shape=(None,1), name='actions')
                self.q=self.build_q_network()
                self.param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/q_network')
                
                with tf.name_scope('q_network_loss'):
                    loss=tf.losses.mean_squared_error(self.target,self.q)
            
                with tf.name_scope('q_network_gradient'):
                    self.gradients=tf.gradients(self.q, self.actions)
            
                with tf.name_scope('train_q_network'):
                    self.train_opt=tf.train.AdamOptimizer(LEARNING_RATE_C).minimize(loss)
            
                with tf.name_scope('update_q_target'):     
                    self.update_opt=[tp.assign(tf.multiply(self.tau,lp)+tf.multiply(1-self.tau,tp)) for tp, lp in zip(self.target_network.param,self.param)]
                    
                with tf.name_scope('initialize_q_target_network'):
                     self.init_target_op=[tp.assign(lp) for tp, lp in zip(self.target_network.param,self.param)]
                     
                                 
    def init_target_network(self):
       self.session.run(self.init_target_op)
             
    def update_target_parameter(self):
        self.session.run(self.update_opt)
        
    def build_target_network(self):
        
        state_action = tf.concat([self.x, self.actions], axis=1)
        
        with tf.variable_scope('q_target_network'):
            
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
    
    def build_q_network(self):
  
        state_action = tf.concat([self.x, self.actions], axis=1)
        
        with tf.variable_scope('q_network'):
            
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
    
    def set_session(self, session):
        self.session=session


class OrnsteinUhlenbeckActionNoise:
    
    def __init__(self, env,mu= 0.0, sigma=0.2, theta=.15, dt=1e-2, x0=None):
        
        self.theta =  theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.initial_noise_scale = 0.1	
        self.noise_decay = 0.99		
        self.noise_process = np.zeros(1)
        
        self.env=env
        
    def get_noise(self, ep):   
        
        noise_scale = (self.initial_noise_scale * self.noise_decay**ep) * (self.env.action_space.high - self.env.action_space.low)
        self.noise_process = self.theta*(self.mu - self.noise_process) + self.sigma*np.random.randn(1)
        
        return noise_scale*self.noise_process


class Model:
    
    def __init__(self):
        
        self.env = gym.make('Pendulum-v0')
        self.state_size = len(self.env.observation_space.sample())
        self.num_episodes=1000
        self.batch_size=64
        
        self.exp_replay=ExperienceReplay(50000,1500)
        
        self.action_noise=OrnsteinUhlenbeckActionNoise(self.env,mu= 0.0, sigma=0.2, theta=.15, dt=1e-2, x0=None)
        
        self.pn_target=PolicyNetwork(scope='target',target_network=None,env=self.env)
        self.p_network=PolicyNetwork(scope='policy',target_network=self.pn_target,env=self.env)
        
        self.qn_target=QNetwork(scope='target',target_network=None,env=self.env)
        self.q_network=QNetwork(scope='q',target_network=self.qn_target,env=self.env)
        
        init = tf.global_variables_initializer()
        self.session = tf.InteractiveSession()
        self.session.run(init)
        
        self.q_network.set_session(self.session)
        self.p_network.set_session(self.session)
        self.pn_target.set_session(self.session)
        self.qn_target.set_session(self.session)
        
        self.q_network.init_target_network()
        self.p_network.init_target_network()
        
    
    def train_networks(self):
        
        if len(self.exp_replay.experience['state']) < self.exp_replay.min_experience:
            return
    
        idx = np.random.choice(len(self.exp_replay.experience['state']), size=BATCH_SIZE, replace=False)
        
        state=np.array([self.exp_replay.experience['state'][i] for i in idx]).reshape(BATCH_SIZE,self.state_size)
        action=np.array([self.exp_replay.experience['action'][i] for i in idx]).reshape(BATCH_SIZE,1)
        reward=[self.exp_replay.experience['reward'][i] for i in idx]
        next_state=np.array([self.exp_replay.experience['next_state'][i] for i in idx]).reshape(BATCH_SIZE,self.state_size)
        dones=[self.exp_replay.experience['done'][i] for i in idx]

        next_actions=self.session.run(self.pn_target.action, feed_dict={self.pn_target.x:next_state})
        
        feed_dict={self.q_network.target_network.x:next_state,
                   self.q_network.target_network.actions:next_actions}
        
        q_next=self.session.run(self.q_network.target_network.q,feed_dict)
     
        targets=np.array([r+0.99*q if not done else r for r, q, done in zip(reward,q_next,dones)])
        #print(targets)
        #train q_network
        feed_dict={self.q_network.x:state, 
                   self.q_network.target:targets, 
                   self.q_network.actions:action
                   }
        self.session.run(self.q_network.train_opt,feed_dict)
        
        
        #train policy_network
        current_actions=self.session.run(self.p_network.action, feed_dict={self.p_network.x:state})
        q_gradient=self.session.run(self.q_network.gradients, feed_dict={self.q_network.x:state, self.q_network.actions:current_actions})
        q_gradient=np.array(q_gradient).reshape(BATCH_SIZE,1)
        
        feed_dict={self.p_network.q_network_gradient: q_gradient,
                   self.p_network.x:state
                   }
        self.session.run(self.p_network.train_opt,feed_dict)
        
        self.p_network.update_target_parameter()
        self.q_network.update_target_parameter()
        

    def playEpisode(self,episode):
        
        state=self.env.reset()
        state=state.reshape(1,self.state_size)
        done=False
        total_reward=0
   
        while not done:

            action=self.p_network.get_action(state)+self.action_noise.get_noise(episode)
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


if __name__ == "__main__":
    pendulum=Model()
    pendulum.run_model()
  
    

        


        
        
        
        
        
        
        
        
        
        
        