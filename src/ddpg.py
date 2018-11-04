import numpy as np
import tensorflow as tf
import gym
from sklearn.kernel_approximation import RBFSampler
import sklearn.pipeline
import sklearn.preprocessing

BATCH_SIZE=64

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
    
    
class Actor:
    
    def __init__(self, env, FLAGS, scope, target_network):
        
        self.env=env
        self.FLAGS=FLAGS
        self.mountainCarEnv=env.get_mountain_env()
        self.tau=1e-2
        
        if scope=='target':
            self.state = tf.placeholder(tf.float32, [None,3], "state")
            self.action = self._action_estimator(scope)
            self.param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/action')
        elif scope=='normal':
            self.target_network=target_network
            self.state = tf.placeholder(tf.float32, [None,3], "state")    
            self.q_gradients=tf.placeholder(dtype=tf.float32, shape=(1,1), name="q_gradients")
            self.param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/action')
            
            with tf.name_scope('init_target'): 
                self.init_target_op=[tp.assign(lp) for tp, lp in zip(target_network.param,self.param)]
            
            with tf.name_scope('update_target'):     
                self.update_target=[tp.assign(tf.multiply(self.tau,lp)+tf.multiply(1-self.tau,tp)) for tp, lp in zip(self.target_network.param,self.param)]
            
            
            self.action = self._action_estimator(scope)
            param=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'action')
            policy_gradient=tf.gradients(self.action, param, -self.q_gradients)
                    
            self.train_op=tf.train.AdamOptimizer(self.FLAGS.learning_rate_Actor).apply_gradients(zip(policy_gradient,param))  
        
    def _action_estimator(self, scope):
        
        with tf.variable_scope('action'):
            W1=tf.get_variable('%sW_1_a'%scope, shape=(3,1), initializer=tf.zeros_initializer)
            a=tf.nn.sigmoid(tf.matmul(self.state, W1))
            a = tf.squeeze(a)
            scalled_actions = self.mountainCarEnv.action_space.low + a*(self.mountainCarEnv.action_space.high - self.mountainCarEnv.action_space.low)
        return scalled_actions  
        
           
    def update(self, state, action, q_gradients):
        #state = self.env.featurize_state(state)
        feed_dict = { self.state: state, self.action: action, self.q_gradients:q_gradients}
        self.session.run(self.train_op, feed_dict)
        

    def get_action(self, state): 
        #state = self.env.featurize_state(state)
        return self.session.run(self.action, feed_dict={self.state:state})

   
    def set_session(self, session):
        self.session=session
    
    def init_target_network(self):
        self.session.run(self.init_target_op)
        
    def update_target_network(self):
        self.session.run(self.update_target)
        
    
class Critic:
    
    def __init__(self, env, FLAGS, scope, target_network):
        
        self.env=env
        self.FLAGS=FLAGS
        self.tau=1e-2
        
        
        if scope=='target':
            
            self.state = tf.placeholder(tf.float32, [None,3], "state")
            self.action = tf.placeholder(tf.float32, [None, 1], "action")
            self.action_value = self._action_value_estimator(scope=scope) 
            self.param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/action_value')
              
        elif scope=='normal':
            
            self.state = tf.placeholder(tf.float32, [None,3], "state")
            self.action = tf.placeholder(tf.float32, [None, 1], "action")
            self.target = tf.placeholder(tf.float32, [None,1], name="target")
            self.target_network=target_network
            self.param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/action_value')
            
            with tf.name_scope('initialize_q_target_network'):
                self.init_target_op=[tp.assign(lp) for tp, lp in zip(self.target_network.param,self.param)]
            
            with tf.name_scope('update_target'):     
                self.update_target=[tp.assign(tf.multiply(self.tau,lp)+tf.multiply(1-self.tau,tp)) for tp, lp in zip(self.target_network.param,self.param)]
                 
            self.action_value=self._action_value_estimator(scope)
            
            self.loss=tf.losses.mean_squared_error(self.target, self.action_value)
    
            self.optimizer = tf.train.AdamOptimizer(self.FLAGS.learning_rate_Critic)
            self.train_op = self.optimizer.minimize(self.loss)  
    
    def _action_value_estimator(self, scope):
        
        state_action = tf.concat([self.state, self.action], axis=1)

        with tf.name_scope(scope+'/action_value'):
            W=tf.get_variable('%s_W_state_value'%scope, shape=(4,1), initializer=tf.zeros_initializer)
            action_value=tf.matmul(state_action, W)
            action_value=tf.squeeze(action_value)         
        return action_value
    
    
    def get_gradients(self, state, action):
        #state = self.env.featurize_state(state)
        return self.session.run(self.compute_gradients, feed_dict={self.state:state, self.action: action})
    
    def set_session(self, session):
        self.session=session
    
    def predict_action_value(self, state, action):
        #state = self.env.featurize_state(state)
        return self.session.run(self.action_value, { self.state: state, self.action:action })
    
    def train(self, state, action, target):
        #state = self.env.featurize_state(state)
        feed_dict = {self.state: state, self.action:action, self.target: target}
        _, loss = self.session.run([self.train_op, self.loss], feed_dict)
        return loss
    
    def init_target_network(self):
        self.session.run(self.init_target_op)
    
    def update_target_network(self):
        self.session.run(self.update_target)
    
    


class Environment:
    
    def __init__(self):
        
        self.env = gym.make('Pendulum-v0')
        self.state_size = len(self.env.observation_space.sample())
   
        observation_examples = np.array([self.env.observation_space.sample() for x in range(10000)])
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.scaler.fit(observation_examples)
 
        self.featurizer = sklearn.pipeline.FeatureUnion([
            ("rbf1", RBFSampler(gamma=5.0, n_components=10)),
            ("rbf2", RBFSampler(gamma=2.0, n_components=10)),
            ("rbf3", RBFSampler(gamma=1.0, n_components=10)),
            ("rbf4", RBFSampler(gamma=0.5, n_components=10))
            ])
        self.featurizer.fit(self.scaler.transform(observation_examples))
    
        
    def get_mountain_env(self):
        return self.env
    
    def get_state_size(self):
        return self.state_size
    
    def featurize_state(self, state):
        """
        Returns the featurized representation for a state.
        """
        scaled = self.scaler.transform(state)
        featurized = self.featurizer.transform(scaled)
        return featurized[0]


class Model:
    
    def __init__(self, FLAGS):
        
        self.env=Environment()
        self.FLAGS=FLAGS
        
        self.actor_target=Actor(self.env, FLAGS, scope='target', target_network=None)
        self.critic_target=Critic(self.env, FLAGS, scope='target', target_network=None)
           
        self.actor=Actor(self.env, FLAGS, scope='normal', target_network=self.actor_target)
        self.critic=Critic(self.env, FLAGS, scope='normal', target_network=self.critic_target)
         
        init = tf.global_variables_initializer()
        session = tf.InteractiveSession()
        session.run(init)
        
        self.critic.set_session(session)
        self.actor.set_session(session)
        self.critic_target.set_session(session)
        self.actor_target.set_session(session)
        
        
        self.actor.init_target_network()
        self.critic.init_target_network()
        
        self.num_episodes=100
        
        self.exp_replay=ExperienceReplay(50000,500)
        
         

    def playEpisode(self,episode):
        
        done=False
        total_reward=0
        iters = 0

        state=self.env.get_mountain_env().reset()
        state=state.reshape(1,self.env.get_state_size())
        
        while not done and iters < 2000:
            
            action=self.actor.get_action(state)
            prev_state=state
            
            state, reward, done, _ = self.env.get_mountain_env().step(action)
            state=state.reshape(1,self.env.get_state_size())
        
            total_reward=total_reward+reward
            
            self.exp_replay.addExperience(prev_state, action, reward, state, done)
            
            self.train_networks()
                      
        return total_reward
            
    def train_networks(self):
        
        if len(self.exp_replay.experience['state']) < self.exp_replay.min_experience:
            return
    
        idx = np.random.choice(len(self.exp_replay.experience['state']), size=BATCH_SIZE, replace=False)
        
        state=np.array([self.exp_replay.experience['state'][i] for i in idx]).reshape(BATCH_SIZE,self.env.get_state_size())
        action=np.array([self.exp_replay.experience['action'][i] for i in idx]).reshape(BATCH_SIZE,1)
        reward=[self.exp_replay.experience['reward'][i] for i in idx]
        next_state=np.array([self.exp_replay.experience['next_state'][i] for i in idx]).reshape(BATCH_SIZE,self.env.get_state_size())
        dones=[self.exp_replay.experience['done'][i] for i in idx]
        
        
        target_actions=self.actor.target_network.get_action(next_state)
        target_actions=np.expand_dims(target_actions,axis=1)
        q_next=self.critic.target_network.predict_action_value(next_state, target_actions)
        
        targets=np.array([r+0.99*q if not done else r for r, q, done in zip(reward,q_next,dones)])
        
        self.critic.train(state, action, np.expand_dims(targets,axis=1))
        
        self.critic.update_target_network()
        self.actor.update_target_network()
        

    def run_model(self):
        
        totalrewards = np.empty(self.num_episodes+1)
        n_steps=1
        
        for n in range(0, self.num_episodes+1):
            
            total_reward=self.playEpisode(n)
            
            totalrewards[n]=total_reward 
            
            if n>0:
                print("episodes: %i, avg_reward (last: %i episodes): %.2f" %(n, n_steps, totalrewards[max(0, n-n_steps):(n+1)].mean()))



tf.app.flags.DEFINE_float('learning_rate_Actor', 0.001, 'Learning rate for the policy estimator')

tf.app.flags.DEFINE_float('learning_rate_Critic', 0.001, 'Learning rate for the state-value estimator')

tf.app.flags.DEFINE_float('gamma', 0.98, 'Future discount factor')

FLAGS = tf.app.flags.FLAGS



if __name__ == "__main__":
    
    pendulum=Model(FLAGS)
    pendulum.run_model()



