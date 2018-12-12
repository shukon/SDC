import logging
import random
import torch
import math


def select_greedy_action(state, policy_net, action_size):
    """ Select the greedy action
    Parameters
    -------
    state: np.array
        state of the environment
    policy_net: torch.nn.Module
        policy network
    action_size: int
        number of possible actions
    Returns
    -------
    int
        ID of selected action
    """
    best_policy = policy_net(state)
    logging.debug("Best policy: %s" % best_policy)
    index = int(best_policy.argmax())
    return index


def select_exploratory_action(state, policy_net, action_size, exploration, t):
    """ Select an action according to an epsilon-greedy exploration strategy
    Parameters
    -------
    state: np.array
        state of the environment
    policy_net: torch.nn.Module
        policy network
    action_size: int
        number of possible actions
    exploration: LinearSchedule
        linear exploration schedule
    t: int
        current time-step
    Returns
    -------
    int
        ID of selected action
    """
    epsilon = exploration.value(t)

    if random.random() <= epsilon:
        act = random.randint(0, action_size - 1)
        logging.debug("Random action: %s" % act)
        return act
    else:
        act = select_greedy_action(state,policy_net, action_size)
        logging.debug("Greedy action: %s" % act)
        return act


def get_action_set():
    """ Get the list of available actions
    Returns
    -------
    list
        list of available actions
    """
    return [[-1.0, 0.05, 0], [1.0, 0.05, 0], [0, 0.5, 0], [0, 0, 1.0]]
