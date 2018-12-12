import gym
import deepq


def main():
    """ 
    Train a Deep Q-Learning agent 
    """ 
    env = gym.make("CarRacing-v0")
    #deepq.learn(env, total_timesteps=1500, learning_starts=100, train_freq=150)
    deepq.learn(env, total_timesteps=10000, learning_starts=100, train_freq=50)
    env.close()


if __name__ == '__main__':
    main()

