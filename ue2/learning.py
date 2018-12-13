import logging

import numpy as np
import torch
import torch.nn
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
    #    1.1 Sample transitions from replay_buffer
    obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = replay_buffer.sample(batch_size)
    obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = torch.tensor(obs_batch), torch.tensor(act_batch), torch.tensor(rew_batch), torch.tensor(next_obs_batch), torch.tensor(done_mask)

    #    1.2 Squeeze observations (add a dimension)

    logging.debug("Shapes: obs_batch=%s, act_batch=%s, rew_batch=%s, next_obs_batch=%s, done_mask=%s" % (
          obs_batch.shape, act_batch.shape, rew_batch.shape, next_obs_batch.shape, done_mask.shape))

    #    2. Compute Q(s_t, a)
    # ASSUMING ACTION IS AN INDEX (squeeze makes ((a, b, c)) to (a, b, c)
    q_batch = policy_net(obs_batch)
    mask = torch.zeros(q_batch.shape).type(torch.ByteTensor)
    for idx, a in enumerate(act_batch):
        mask[idx][a] = 1
    q_batch = torch.masked_select(q_batch, mask)

    #    3. Compute \max_a Q(s_{t+1}, a) for all next states.
    q_next_batch = target_net(obs_batch)
    q_next_batch = torch.max(q_next_batch, 1)[0]

    #    4. Mask next state values where episodes have terminated
    # Following nature-paper page 7 Algorithm 1 this means replacing q_next_batch with rewards (so zero here) for terminations
    done_mask = done_mask.type(torch.ByteTensor)
    q_next_batch.masked_fill(done_mask, 0)

    #    5. Compute the target
    q_next_batch *= gamma
    target = rew_batch + q_next_batch

    # Reset gradients
    optimizer.zero_grad()

    #    6. Compute the loss
    logging.debug("Targets: %s" % target[:5])
    criterion = torch.nn.MSELoss()
    loss = criterion(target, q_batch)

    #    7. Calculate the gradients
    grad = loss.backward()

    #    8. Clip the gradients
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 1)

    #    9. Optimize the model
    optimizer.step()

    return loss.item()

def update_target_net(policy_net, target_net):
    """ Update the target network
    Parameters
    -------
    policy_net: torch.nn.Module
        policy Q-network
    target_net: torch.nn.Module
        target Q-network
    """
    target_net.load_state_dict(policy_net.state_dict())
