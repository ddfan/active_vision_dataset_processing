#!/usr/bin/env python3

import os
import gym
import gym_activevision
import argparse
import numpy as np
from baselines.a2c.a2c import play
from baselines.a2c.utils import fc,batch_to_seq,seq_to_batch,lstm
from baselines.a2c.policies import nature_cnn
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

MODEL_NAME = '/home/david/Documents/active_vision_dataset_processing/logs/openai-2018-02-27-01-44-18-124640/checkpoint05500'

from train_activevision_a2c import make_activevision_env,MobilenetPolicy,LstmMobilenetPolicy

def playpolicy(env_id, num_timesteps, seed, policy, num_env,load_path,render):
    if policy == 'cnn':
        policy_fn = CnnPolicy
    elif policy == 'lstm':
        policy_fn = LstmPolicy
    elif policy == 'lnlstm':
        policy_fn = LnLstmPolicy
    elif policy == 'mobilenet':
        policy_fn = MobilenetPolicy
    elif policy == 'lstmmobilenet':
        policy_fn= LstmMobilenetPolicy
    #env = VecFrameStack(make_atari_env(env_id, num_env, seed), 4)
    env = make_activevision_env(env_id,num_env,seed)
    play(policy_fn, env, seed, total_timesteps=num_timesteps, load_path=load_path, render=render)
    env.close()

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm', 'mobilenet','lstmmobilenet'], default='lstmmobilenet')
    parser.add_argument('--env', help='environment ID', default='ActiveVision-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e4))
    parser.add_argument('--load_path', help='path to model directory', default=MODEL_NAME)
    parser.add_argument('--render', help='Render when playing? (1/0)', type=bool, default=True)
    args = parser.parse_args()
    logger.configure()
    playpolicy(args.env, num_timesteps=args.num_timesteps, seed=args.seed,
        policy=args.policy, num_env=1, render=args.render, load_path=args.load_path)

if __name__ == '__main__':
    main()
