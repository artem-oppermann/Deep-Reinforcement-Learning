import numpy as np
import tensorflow as tf
import gym
from sklearn.kernel_approximation import RBFSampler
import sklearn.pipeline
import sklearn.preprocessing
import imageio


class Actor:
    
    def __init__(self, env, FLAGS):
        
        self.env=env
        self.FLAGS=FLAGS
        mountainCarEnv=env.get_mountain_env()
        self.state = tf.placeholder(tf.float32, [40], "state")
        self.td_error = tf.placeholder(dtype=tf.float32, name="td_error")
        
        self.mu = self._mu_classifier()
        self.sigma = self._sigma_classifier()
        
        self.normal_dist = tf.contrib.distributions.Normal(self.mu, self.sigma)
        self.action = self.normal_dist._sample_n(1)
        self.action = tf.clip_by_value(self.action, mountainCarEnv.action_space.low[0], mountainCarEnv.action_space.high[0])
        
        # Loss and train op
        self.loss = -self.normal_dist.log_prob(self.action) * self.td_error
                                              
        # Add cross entropy cost to encourage exploration
        self.loss -= 1e-1 * self.normal_dist.entropy()
        
        self.optimizer = tf.train.AdamOptimizer(self.FLAGS.learning_rate_Actor)
        self.train_op = self.optimizer.minimize(self.loss)
    
    def _mu_classifier(self):
        
        with tf.name_scope('mu'):
            W1=tf.get_variable('W_1_mu', shape=(40,1), initializer=tf.zeros_initializer)
            mu=tf.matmul(tf.expand_dims(self.state, 0), W1)
            mu = tf.squeeze(mu)
        return mu  
    
    def _sigma_classifier(self):
        
        with tf.name_scope('sigma'):
            W2=tf.get_variable('W_1_sigma', shape=(40,1), initializer=tf.zeros_initializer)
            sigma=tf.matmul(tf.expand_dims(self.state, 0), W2)
            sigma = tf.squeeze(sigma)
            sigma = tf.nn.softplus(sigma) + 1e-5                 
        return sigma
           
    def update(self, state, td_error, action):
        
        state = self.env.featurize_state(state)
        feed_dict = { self.state: state, self.td_error: td_error, self.action: action  }
        _, loss = self.session.run([self.train_op, self.loss], feed_dict)
        return loss

    def sample_action(self, state):   
        state = self.env.featurize_state(state)
        return self.session.run(self.action, feed_dict={self.state:state})

   
    def set_session(self, session):
        self.session=session
    
    
class Critic:
    
    def __init__(self, env, FLAGS):
        
        self.env=env
        self.FLAGS=FLAGS
        
        self.state = tf.placeholder(tf.float32, [40], "state")
        self.target = tf.placeholder(dtype=tf.float32, name="target")

        self.state_value = self._value_estimator()
        self.loss = tf.squared_difference(self.state_value, self.target)

        self.optimizer = tf.train.AdamOptimizer(self.FLAGS.learning_rate_Critic)
        self.train_op = self.optimizer.minimize(self.loss)  
    
    def _value_estimator(self):
        
        with tf.name_scope('state_value'):
            W=tf.get_variable('W_state_value', shape=(40,1), initializer=tf.zeros_initializer)
            state_value=tf.matmul(tf.expand_dims(self.state, 0), W)
            state_value=tf.squeeze(state_value)
        return state_value
    
    def set_session(self, session):
        self.session=session
    
    def predict(self, state):
        state = self.env.featurize_state(state)
        return self.session.run(self.state_value, { self.state: state })
    
    def update(self, state, target):
        state = self.env.featurize_state(state)
        feed_dict = { self.state: state, self.target: target }
        _, loss = self.session.run([self.train_op, self.loss], feed_dict)
        return loss
    
    


class Environment:
    
    def __init__(self):
        
        self.env = gym.make('MountainCarContinuous-v0')
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
           
        self.actor=Actor(self.env, FLAGS)
        self.critic=Critic(self.env, FLAGS)
            
        init = tf.global_variables_initializer()
        session = tf.InteractiveSession()
        session.run(init)
        
        self.critic.set_session(session)
        self.actor.set_session(session)
        
        self.num_episodes=100
        
         

    def playEpisode(self,episode):
        
        state=self.env.get_mountain_env().reset()
        state=state.reshape(1,self.env.get_state_size())
        
        done=False
        total_reward=0
        iters = 0
        
        images=[]

        while not done and iters < 2000:
            
            action=self.actor.sample_action(state)
            prev_state=state
            state, reward, done, _ = self.env.get_mountain_env().step(action)
            state=state.reshape(1,self.env.get_state_size())
            
            if episode>10:
                image=self.env.get_mountain_env().render(mode='rgb_array')
                images.append(image)
            
            total_reward=total_reward+reward
            
            # Calculate TD Target
            value_next = self.critic.predict(state)
            td_target = reward + self.FLAGS.gamma * value_next
            td_error = td_target - self.critic.predict(prev_state)
           
            # Update the value estimator
            self.critic.update(prev_state, td_target)
            
            # Update the policy estimator
            self.actor.update(state, td_error, action)
        
        if episode>10:
            imageio.mimsave("C:/Users/Admin/Desktop/Deep Learning/" + 'mountain_car_%i.gif'%episode, images, fps=30)
            
            
        return total_reward
            

    def run_model(self):
        
        totalrewards = np.empty(self.num_episodes+1)
        n_steps=1
        
        for n in range(0, self.num_episodes+1):
            
            total_reward=self.playEpisode(n)
            
            totalrewards[n]=total_reward 
            
            if n>0:
                print("episodes: %i, avg_reward (last: %i episodes): %.2f" %(n, n_steps, totalrewards[max(0, n-n_steps):(n+1)].mean()))



tf.app.flags.DEFINE_float('learning_rate_Actor', 0.001, 'Learning rate for the policy estimator')

tf.app.flags.DEFINE_float('learning_rate_Critic', 0.1, 'Learning rate for the state-value estimator')

tf.app.flags.DEFINE_float('gamma', 0.95, 'Future discount factor')

FLAGS = tf.app.flags.FLAGS



if __name__ == "__main__":
    
    pendulum=Model(FLAGS)
    pendulum.run_model()
    






