import gym
import deepq


def main():
    """ 
    Evaluate a trained Deep Q-Learning agent 
    """ 
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='agent')
    args = parser.parse_args()
    print(args)

    env = gym.make("CarRacing-v0")
    deepq.evaluate(env, load_path=args.path + '.pt')
    env.close()

if __name__ == '__main__':
    main()
