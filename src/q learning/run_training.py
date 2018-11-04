
from cartpole_env import CartPole
import tensorflow as tf


tf.app.flags.DEFINE_integer('batch_size', 64, 'Batch size.')

tf.app.flags.DEFINE_float('learning_rate',5e-4,'Learning_Rate')

tf.app.flags.DEFINE_float('gamma', 0.99, 'Future discount factor')

tf.app.flags.DEFINE_integer('num_iter_update', 25,'Number of iterations after the target network gets updated.')

FLAGS = tf.app.flags.FLAGS


if __name__ == "__main__":
    
    cartPole=CartPole(FLAGS)
    cartPole.run()