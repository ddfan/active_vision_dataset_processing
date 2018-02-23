#!/usr/bin/env python3

import os
import gym
import gym_activevision
import argparse
from baselines.a2c.a2c import learn
from baselines.a2c.utils import fc
from baselines import logger
from baselines.bench import Monitor
from baselines.common import misc_util
from baselines.common import set_global_seeds
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.distributions import make_pdtype
from baselines.common.tf_util import load_img
import tensorflow as tf
import tensorflow.contrib.layers as layers
import mobilenet_v1

MODEL_NAME = './checkpoints/mobilenet_v1_1.0_224'

def make_activevision_env(env_id, num_env, seed, wrapper_kwargs=None, start_index=0):
    """
    Create a wrapped, monitored SubprocVecEnv for ActiveVision.
    """
    def make_env(rank): # pylint: disable=C0111
        def _thunk():
            env = gym.make(env_id)
            env.seed(seed + rank)
            env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
            return env
        return _thunk
    set_global_seeds(seed)
    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])

class MobilenetPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False): #pylint: disable=W0613
            #setup input placeholder
            nh, nw, nc = ob_space.shape
            ob_shape = (nbatch, nh, nw, nc)
            nact = ac_space.n
            X = tf.placeholder(tf.float32, ob_shape) #obs

            #graph for mobilenet + fully connected layers
            with tf.variable_scope("model", reuse=reuse):
                this_scope=tf.get_default_graph().get_name_scope()
                #load mobilenet
                arg_scope = mobilenet_v1.mobilenet_v1_arg_scope(weight_decay=0.0)
                with tf.contrib.slim.arg_scope(arg_scope):
                    logits, _ = mobilenet_v1.mobilenet_v1(X,num_classes=1001,is_training=False,depth_multiplier=1.0)

                conv_out = tf.squeeze(tf.get_default_graph().get_tensor_by_name(this_scope+"/MobilenetV1/Logits/AvgPool_1a/AvgPool:0"),[1,2])
                conv_out=tf.stop_gradient(conv_out)
                #conv_out=tf.Print(conv_out,[tf.reduce_min(conv_out),tf.reduce_max(conv_out)])

                #restore variables for mobilenet
                rest_var = tf.contrib.slim.get_variables_to_restore()
                var_dict={}
                for var in rest_var:
                    if var.name.startswith(this_scope+'/MobilenetV1'):
                        noscope_name=var.name.replace(this_scope+'/','')
                        noscope_name=noscope_name.replace(':0','')
                        var_dict[noscope_name]=var      
                if var_dict:
                    saver = tf.train.Saver(var_dict)
                    saver.restore(sess, MODEL_NAME+'.ckpt')

                #grab first pixel and reinterpret as target
                #make onehot vector and append
                target=tf.cast(X[:,0,0,0],tf.int32)
                target_vec=tf.one_hot(target,34,axis=-1)
                h = tf.concat([conv_out,target_vec],axis=1)
                
                #policy and value function layers
                pi = fc(h, 'pi', nact, init_scale=0.01)
                vf = fc(h, 'v', 1)[:,0]

            # begin probability density layer
            self.pdtype = make_pdtype(ac_space)
            self.pd = self.pdtype.pdfromflat(pi)

            a0 = self.pd.sample()
            neglogp0 = self.pd.neglogp(a0)
            self.initial_state = None

            def step(ob, *_args, **_kwargs):
                ob=load_img(ob)
                a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob})
                return a, v, self.initial_state, neglogp

            def value(ob, *_args, **_kwargs):
                ob=load_img(ob)
                return sess.run(vf, {X:ob})

            self.X = X
            self.pi = pi
            self.vf = vf
            self.step = step
            self.value = value

def train(env_id, num_timesteps, seed, policy, lrschedule, num_env):
    if policy == 'cnn':
        policy_fn = CnnPolicy
    elif policy == 'lstm':
        policy_fn = LstmPolicy
    elif policy == 'lnlstm':
        policy_fn = LnLstmPolicy
    elif policy == 'mobilenet':
        policy_fn = MobilenetPolicy
    #env = VecFrameStack(make_atari_env(env_id, num_env, seed), 4)
    env = make_activevision_env(env_id,num_env,seed)
    learn(policy_fn, env, seed, total_timesteps=int(num_timesteps * 1.1), lrschedule=lrschedule, save_interval=1000)
    env.close()

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm', 'mobilenet'], default='mobilenet')
    parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear'], default='constant')
    parser.add_argument('--env', help='environment ID', default='ActiveVision-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(50e6))
    args = parser.parse_args()
    logger.configure()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed,
        policy=args.policy, lrschedule=args.lrschedule, num_env=32)

if __name__ == '__main__':
    main()
