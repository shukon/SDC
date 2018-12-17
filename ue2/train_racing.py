import gym
import deepq


def main():
    """ 
    Train a Deep Q-Learning agent 
    """ 
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_fn', type=str, default='agent')
    args = parser.parse_args()
    print(args)

    env = gym.make("CarRacing-v0")
    #deepq.learn(env, total_timesteps=1500, learning_starts=100, train_freq=150)
    deepq.learn(env, total_timesteps=10000, learning_starts=100, train_freq=50,
                model_identifier=args.output_fn,
                )
    env.close()


if __name__ == '__main__':
    main()

