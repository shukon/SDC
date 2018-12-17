import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, action_size, device):
        """ Create Q-network
        Parameters
        ----------
        action_size: int
            number of actions
        device: torch.device
            device on which to the model will be allocated
        """
        super().__init__()

        self.device = device
        self.action_size = action_size

        self.use_sensor = True

        self.conv1 = nn.Conv2d(3, 32, 9)
        self.conv2 = nn.Conv2d(32, 10, 5)
        if self.use_sensor:
            self.fc1 = nn.Linear(16 * 10 * 5 * 5 + 7, 120)
        else:
            self.fc1 = nn.Linear(16 * 10 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, action_size)

    def forward(self, observation):
        """ Forward pass to compute Q-values
        Parameters
        ----------
        observation: np.array
            array of state(s)
        Returns
        ----------
        torch.Tensor
            Q-values
        """
        # Perform convolution on the image-part of the input
        x = torch.tensor(observation).permute([0,3,1,2])
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        # Make one flat tensor out of x
        x = x.view(-1, self._num_flat_features(x))
        # Add sensordata
        if self.use_sensor:
            speed, abs_sensors, steering, gyroscope = self.extract_sensor_values(observation, x.shape[0])
            x = torch.cat((x, speed, abs_sensors, steering, gyroscope), 1)

        # Second part of network (without convolution)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        #x = F.softmax(x)
        return x

    def _num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


    def extract_sensor_values(self, observation, batch_size):
        """ Extract numeric sensor values from state pixels
        Parameters
        ----------
        observation: list
            python list of batch_size many torch.Tensors of size (96, 96, 3)
        batch_size: int
            size of the batch
        Returns
        ----------
        torch.Tensors of size (batch_size, 1),
        torch.Tensors of size (batch_size, 4),
        torch.Tensors of size (batch_size, 1),
        torch.Tensors of size (batch_size, 1)
            Extracted numerical values
        """
        observation = torch.tensor(observation)
        speed_crop = observation[:, 84:94, 12, 0].reshape(batch_size, -1)
        speed = speed_crop.sum(dim=1, keepdim=True) / 255
        abs_crop = observation[:, 84:94, 18:25:2, 2].reshape(batch_size, 10, 4)
        abs_sensors = abs_crop.sum(dim=1) / 255
        steer_crop = observation[:, 88, 38:58, 1].reshape(batch_size, -1)
        steering = steer_crop.sum(dim=1, keepdim=True)
        gyro_crop = observation[:, 88, 58:86, 0].reshape(batch_size, -1)
        gyroscope = gyro_crop.sum(dim=1, keepdim=True)
        return speed, abs_sensors.reshape(batch_size, 4), steering, gyroscope
