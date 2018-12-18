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

        self.softmax = False

        conv1_filter_size = 16
        conv2_filter_size = 8
        conv1_size = 16
        conv2_size = 32

        denses = [256, 64, # 120, 52, ... add more dense layers by writing sizes here...
                  action_size]
        self.fcs = []  # dense forward layers

        self.conv1 = nn.Conv2d(3, conv1_size, conv1_filter_size, stride=2)
        self.conv2 = nn.Conv2d(conv1_size, conv2_size, conv2_filter_size, stride=2)
        if self.use_sensor:
            self.fcs.append(nn.Linear(16 * conv2_size * conv2_filter_size * conv2_filter_size + 7, denses[0]))
        else:
            self.fcs.append(nn.Linear(16 * conv2_size * conv2_filter_size * conv2_filter_size, denses[0]))
        for in_size, out_size in zip(denses[:-1], denses[1:]):
            self.fcs.append(nn.Linear(in_size, out_size))

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
        x = torch.tensor(observation).permute([0, 3, 1, 2])
        x = F.max_pool2d(F.relu(self.conv1(x)), 5)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        # Make one flat tensor out of x
        x = x.view(-1, self._num_flat_features(x))
        # Add sensordata
        if self.use_sensor:
            speed, abs_sensors, steering, gyroscope = self.extract_sensor_values(observation, x.shape[0])
            x = torch.cat((x, speed, abs_sensors, steering, gyroscope), 1)

        # Second part of network (without convolution)
        for fc in self.fcs[:-1]:
            x = F.relu(fc(x))
        # Last one without relu (just linear) if not softmax
        if self.softmax:
            x = F.relu(self.fcs[-1](x))
            x = F.softmax(x)
        else:
            x = self.fcs[-1](x)

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
