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
    parser.add_argument('--num-timesteps', type=int, default=int(1000000))
    args = parser.parse_args()
    logger.configure()
    set_global_seeds(args.seed)
    env = gym.make(args.env)
    env = bench.Monitor(env, logger.get_dir())
    model = deepq.models.cnn_to_mlp_activevision(
        convs=[(32, 16, 8), (64, 8, 4), (64, 4, 2)],
        hiddens=[256],  
        dueling=bool(args.dueling),
    )
    act = deepq.learn(
        env,
        q_func=model,
        lr=1e-4,
        max_timesteps=args.num_timesteps,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        train_freq=5,
        learning_starts=50000,
        target_network_update_freq=100,
        gamma=0.99,
        prioritized_replay=bool(args.prioritized),
        print_freq=100,
        batch_size=32
    )
    print("Saving model to activevision_model.pkl")
    act.save("activevision_model.pkl")
    env.close()

if __name__ == '__main__':
    main()
