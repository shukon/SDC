import numpy as np
import torch
import torch.nn.functional as F


def perform_qlearning_step(policy_net, target_net, optimizer, replay_buffer, batch_size, gamma, device):
    """ Perform a deep Q-learning step
    Parameters
    -------
    policy_net: torch.nn.Module
        policy Q-network
    target_net: torch.nn.Module
        target Q-network
    optimizer: torch.optim.Adam
        optimizer
    replay_buffer: ReplayBuffer
        replay memory storing transitions
    batch_size: int
        size of batch to sample from replay memory 
    gamma: float
        discount factor used in Q-learning update
    device: torch.device
        device on which to the models are allocated
    Returns
    -------
    float
        loss value for current learning step
    """
    #    1. Sample transitions from replay_buffer
    obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = replay_buffer.sample(batch_size)

    #    2. Compute Q(s_t, a)
    # ASSUMING ACTION IS AN INDEX
    q_batch = np.array([policy_net(obs)[act] for obs, act in zip(obs_batch, act_batch)])

    #    3. Compute \max_a Q(s_{t+1}, a) for all next states.
    q_next_batch = np.array([max(target_net(obs)) for obs in next_obs_batch])

    #    4. Mask next state values where episodes have terminated
    # Following nature-paper page 7 Algorithm 1 this means replacing q_next_batch with rewards (so zero here) for terminations
    q_next_batch = np.array([q if d == 0 else 0 for q, d in zip(q_next_batch, done_mask)])

    #    5. Compute the target
    target = np.array([r + gamma * q for r, q in zip(rew_batch, q_next_batch)])

    #    6. Compute the loss
    loss = torch.mean([(t - q)**2 for t, q in zip(target, q_batch)])

    #    7. Calculate the gradients
    #    8. Clip the gradients
    #    9. Optimize the model
    # TODO: Run single Q-learning step

def update_target_net(policy_net, target_net):
    """ Update the target network
    Parameters
    -------
    policy_net: torch.nn.Module
        policy Q-network
    target_net: torch.nn.Module
        target Q-network
    """

    # TODO: Update target network
