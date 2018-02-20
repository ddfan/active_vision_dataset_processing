import gym
import gym_activevision
from baselines import deepq
from baselines.common import set_global_seeds
from baselines import bench
import argparse
from baselines import logger
from baselines.common import misc_util
import tensorflow as tf
import tensorflow.contrib.layers as layers

import mobilenet_v1

MODEL_NAME = './checkpoints/mobilenet_v1_1.0_224'
img_size=224
factor=1.0
num_classes=1001
is_training=False
weight_decay = 0.0


def cnn_to_mlp_mobilenet(hiddens, layer_norm=False):
	"""This model takes as input an observation and returns values of all actions.

	Parameters
	----------
	convs: [(int, int int)]
		list of convolutional layers in form of
		(num_outputs, kernel_size, stride)
	hiddens: [int]
		list of sizes of hidden layers
	dueling: bool
		if true double the output MLP to compute a baseline
		for action scores

	Returns
	-------
	q_func: function
		q_function for DQN algorithm.
	"""

	return lambda *args, **kwargs: _cnn_to_mlp_mobilenet(hiddens, layer_norm=layer_norm, *args, **kwargs)

def _cnn_to_mlp_mobilenet(hiddens, inpt, num_actions, scope, reuse=False, layer_norm=False):
	with tf.variable_scope(scope, reuse=reuse):
		with tf.variable_scope("convnet"):
			inp = inpt[:,:,:,0:3] #ignore depth for now
			this_scope=tf.get_default_graph().get_name_scope()
			#load mobilenet
			arg_scope = mobilenet_v1.mobilenet_v1_arg_scope(weight_decay=weight_decay)
			with tf.contrib.slim.arg_scope(arg_scope):
				logits, _ = mobilenet_v1.mobilenet_v1(inp,num_classes=num_classes,is_training=is_training,depth_multiplier=factor)

			conv_out = tf.squeeze(tf.get_default_graph().get_tensor_by_name(this_scope+"/MobilenetV1/Logits/AvgPool_1a/AvgPool:0"),[1,2])
			conv_out=tf.stop_gradient(conv_out)
			#conv_out=tf.Print(conv_out,[tf.reduce_min(conv_out),tf.reduce_max(conv_out)])
			rest_var = tf.contrib.slim.get_variables_to_restore()
			var_dict={}
			for var in rest_var:
				if var.name.startswith(this_scope+'/MobilenetV1'):
					noscope_name=var.name.replace(this_scope+'/','')
					noscope_name=noscope_name.replace(':0','')
					var_dict[noscope_name]=var		

			if var_dict:
				saver = tf.train.Saver(var_dict)
				saver.restore(tf.get_default_session(), MODEL_NAME+'.ckpt')

		with tf.variable_scope("action_value"):
			target=tf.cast(inpt[:,0,0,0],tf.int32)
			target_vec=tf.one_hot(target,34,axis=-1)
			action_out = tf.concat([conv_out,target_vec],axis=1)
			for hidden in hiddens:
				action_out = layers.fully_connected(action_out, num_outputs=hidden, activation_fn=None)
				if layer_norm:
					action_out = layers.layer_norm(action_out, center=True, scale=True)
				action_out = tf.nn.relu(action_out)
			action_scores = layers.fully_connected(action_out, num_outputs=num_actions, activation_fn=None)

		return action_scores


def main():
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--env', help='environment ID', default='ActiveVision-v0')
	parser.add_argument('--seed', help='RNG seed', type=int, default=0)
	parser.add_argument('--prioritized', type=int, default=1)
	parser.add_argument('--dueling', type=int, default=0)
	parser.add_argument('--num-timesteps', type=int, default=int(1000000))
	args = parser.parse_args()
	logger.configure()
	set_global_seeds(args.seed)
	env = gym.make(args.env)
	env = bench.Monitor(env, logger.get_dir())
	model = cnn_to_mlp_mobilenet(hiddens=[256])
	act = deepq.learn(
		env,
		q_func=model,
		lr=1e-4,
		max_timesteps=args.num_timesteps,
		buffer_size=50000,
		exploration_fraction=0.1,
		exploration_final_eps=0.01,
		train_freq=5,
		learning_starts=10000,
		target_network_update_freq=100,
		gamma=0.99,
		prioritized_replay=bool(args.prioritized),
		print_freq=100,
		batch_size=64
	)
	print("Saving model to activevision_model.pkl")
	act.save("activevision_model.pkl")
	env.close()

if __name__ == '__main__':
	main()

