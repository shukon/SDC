from pyvirtualdisplay import Display
import gym
import deepq


def main():
    """ 
    Train a Deep Q-Learning agent in headless mode on the cluster
    """ 
    display = Display(visible=0, size=(800,600))
    display.start()
    env = gym.make("CarRacing-v0")
    #deepq.learn(env)
    deepq.learn(env,  #, total_timesteps=10000, learning_starts=100, train_freq=50,)
		model_identifier='cluster_agent')
    env.close()
    display.stop()


if __name__ == '__main__':
    main()

