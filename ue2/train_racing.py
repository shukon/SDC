import gym
import deepq


def main():
    """ 
    Train a Deep Q-Learning agent 
    """ 
    env = gym.make("CarRacing-v0")
    deepq.learn(env, total_timesteps=100, learning_starts=10, train_freq=30)
    env.close()


if __name__ == '__main__':
    main()

