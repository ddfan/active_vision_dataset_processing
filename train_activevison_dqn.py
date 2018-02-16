import gym
import gym_activevision
from baselines import deepq
from baselines.common import set_global_seeds
from baselines import bench
import argparse
from baselines import logger
from baselines.common import misc_util

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='ActiveVision-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--prioritized', type=int, default=1)
    parser.add_argument('--dueling', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(10000))
    args = parser.parse_args()
    logger.configure()
    set_global_seeds(args.seed)
    env = gym.make(args.env)
    env = bench.Monitor(env, logger.get_dir())
    model = deepq.models.cnn_to_mlp(
        convs=[(32, 8, 128), (64, 4, 2), (64, 3, 1)],
        hiddens=[256],
        dueling=bool(args.dueling),
    )
    act = deepq.learn(
        env,
        q_func=model,
        lr=1e-4,
        max_timesteps=args.num_timesteps,
        buffer_size=1000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        train_freq=10,
        learning_starts=1000,
        target_network_update_freq=100,
        gamma=0.99,
        prioritized_replay=bool(args.prioritized),
        print_freq=10
    )
    print("Saving model to activevision_model.pkl")
    act.save("activevision_model.pkl")
    env.close()

if __name__ == '__main__':
    main()
