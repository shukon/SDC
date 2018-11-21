import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationNetwork(torch.nn.Module):
    def __init__(self):
        """
        1.1 d)
        Implementation of the network layers. The image size of the input
        observations is 96x96 pixels.
        """
        super().__init__()
        gpu = torch.device('cuda')

        self.use_sensor = True

        #view (****) this please
        self.action2name = {
                torch.tensor([-1.0, 0.0, 0.8]) : 'steer_left_brake',
                torch.tensor([ 1.0, 0.0, 0.0]) : 'steer_right',
                torch.tensor([ 1.0, 0.5, 0.0]) : 'steer_right',
                torch.tensor([ 1.0, 0.0, 0.8]) : 'steer_right_brake',
                torch.tensor([ 0.0, 0.5, 0.0]) : 'gas',
                torch.tensor([ 0.0, 0.1, 0.0]) : 'chill',
                torch.tensor([-1.0, 0.5, 0.0]) : 'steer_left',
                torch.tensor([ 0.0, 0.0, 0.8]) : 'brake',
                torch.tensor([-1.0, 0.0, 0.0]) : 'steer_left'}
        self.name2actions = {v : k for k, v in self.action2name.items()}
        self.onehot2name = {
                   torch.tensor([1, 0, 0, 0, 0, 0, 0]) : 'steer_left',
                   torch.tensor([0, 1, 0, 0, 0, 0, 0]) : 'steer_right',
                   torch.tensor([0, 0, 1, 0, 0, 0, 0]) : 'steer_left_brake',
                   torch.tensor([0, 0, 0, 1, 0, 0, 0]) : 'steer_right_brake',
                   torch.tensor([0, 0, 0, 0, 1, 0, 0]) : 'brake',
                   torch.tensor([0, 0, 0, 0, 0, 1, 0]) : 'gas',
                   torch.tensor([0, 0, 0, 0, 0, 0, 1]) : 'chill'
                   }
        self.name2onehot = {v : k for k, v in self.onehot2name.items()}

        # Network

        self.conv1 = nn.Conv2d(3, 32, 9)
        self.conv2 = nn.Conv2d(32, 10, 5)
    # an affine operation: y = Wx + b
        if self.use_sensor:
            self.fc1 = nn.Linear(16 * 10 * 5 * 5 + 7, 120)
        else:
            self.fc1 = nn.Linear(16 * 10 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 7)

    def forward(self, observation):
        """
        1.1 e)
        The forward pass of the network. Returns the prediction for the given
        input observation.
        observation:   torch.Tensor of size (batch_size, 96, 96, 3)
        return         torch.Tensor of size (batch_size, number_of_classes)
        """
        x = observation.permute([0,3,1,2])
        # Max pooling over a (2, 2) windowgit
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self._num_flat_features(x))
        if self.use_sensor:
            speed, abs_sensors, steering, gyroscope = self.extract_sensor_values(observation, x.shape[0])
            x = torch.cat((x, speed, abs_sensors, steering, gyroscope), 1)
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

    def actions_to_classes(self, actions):
        """
        1.1 c)
        For a given set of actions map every action to its corresponding
        action-class representation. Assume there are number_of_classes
        different classes, then every action is represented by a
        number_of_classes-dim vector which has exactly one non-zero entry
        (one-hot encoding). That index corresponds to the class number.
        actions:        python list of N torch.Tensors of size 3
        return          python list of N torch.Tensors of size number_of_classes
        """
        #(****) marks the order of these:
        listofclasses = []
        for a in actions:
            if(torch.all(torch.eq(a,torch.tensor([-1.0, 0.0, 0.8])))):
                listofclasses.append(torch.tensor([0, 0, 1, 0, 0, 0, 0]))
            elif(torch.all(torch.eq(a,torch.tensor([ 1.0, 0.5, 0.0])))):
                listofclasses.append(torch.tensor([0, 1, 0, 0, 0, 0, 0]))
            elif(torch.all(torch.eq(a,torch.tensor([ 1.0, 0.0, 0.0])))):
                listofclasses.append(torch.tensor([0, 1, 0, 0, 0, 0, 0]))
            elif(torch.all(torch.eq(a,torch.tensor([ 1.0, 0.0, 0.8])))):
                listofclasses.append(torch.tensor([0, 0, 0, 1, 0, 0, 0]))
            elif(torch.all(torch.eq(a,torch.tensor([ 0.0, 0.5, 0.0])))):
                listofclasses.append(torch.tensor([0, 0, 0, 0, 0, 1, 0]))
            elif(torch.all(torch.eq(a,torch.tensor([ 0.0, 0.0, 0.0])))):
                listofclasses.append(torch.tensor([0, 0, 0, 0, 0, 0, 1]))
            elif(torch.all(torch.eq(a,torch.tensor([-1.0, 0.5, 0.0])))):
                listofclasses.append(torch.tensor([1, 0, 0, 0, 0, 0, 0]))
            elif(torch.all(torch.eq(a,torch.tensor([ 0.0, 0.0, 0.8])))):
                listofclasses.append(torch.tensor([0, 1, 0, 0, 1, 0, 0]))
            elif(torch.all(torch.eq(a,torch.tensor([-1.0, 0.0, 0.0])))):
                listofclasses.append(torch.tensor([1, 0, 0, 0, 0, 0, 0]))
            else: listofclasses.append(torch.tensor([0, 0, 0, 0, 0, 0, 1]))

        return listofclasses
        #actions = [[self.action2name[str(t)] for t in a] for a in actions]
        #naureturn [self.name2onehot[a] for a in actions]


    def scores_to_action(self, scores):
        """
        1.1 c)
        Maps the scores predicted by the network to an action-class and returns
        the corresponding action [steer, gas, brake].
        scores:         python list of torch.Tensors of size number_of_classes
        return          (float, float, float)
        """
        #max_index = torch.max(scores, 1)[1]
        #onehot = torch.zeros(7)
        #onehot[max_index] = 1
        #act = self.name2actions[self.onehot2name[onehot]]
        max_idx = torch.max(scores)
        classtensor = (scores == max_idx)
        classtensor = classtensor.type(torch.long)
        if(torch.all(torch.eq(classtensor,torch.tensor([1, 0, 0, 0, 0, 0, 0])))):
            return (-1.0, 0.0, 0.0) #steer_left
        elif(torch.all(torch.eq(classtensor,torch.tensor([0, 1, 0, 0, 0, 0, 0])))):
            return (1.0, 0.0, 0.0) #steer_right
        elif(torch.all(torch.eq(classtensor,torch.tensor([0, 0, 1, 0, 0, 0, 0])))):
            return (-1.0, 0.0, 0.8) #steer_left_brake
        elif(torch.all(torch.eq(classtensor,torch.tensor([0, 0, 0, 1, 0, 0, 0])))):
            return (1.0, 0.0, 0.8) #steer_right_brake
        elif(torch.all(torch.eq(classtensor,torch.tensor([0, 0, 0, 0, 1, 0, 0])))):
            return (0.0, 0.0, 0.8) #brake
        elif(torch.all(torch.eq(classtensor,torch.tensor([0, 0, 0, 0, 0, 1, 0])))):
            return (0.0, 0.5, 0.0) #gas
        elif(torch.all(torch.eq(classtensor,torch.tensor([0, 0, 0, 0, 0, 0, 1])))):
            return (0.0, 0.0, 0.0) #chill brah

        return (0.0, 0.0, 0.8)


    def extract_sensor_values(self, observation, batch_size):
        """
        observation:    python list of batch_size many torch.Tensors of size
                        (96, 96, 3)
        batch_size:     int
        return          torch.Tensors of size (batch_size, 1),
                        torch.Tensors of size (batch_size, 4),
                        torch.Tensors of size (batch_size, 1),
                        torch.Tensors of size (batch_size, 1)
        """
        speed_crop = observation[:, 84:94, 12, 0].reshape(batch_size, -1)
        speed = speed_crop.sum(dim=1, keepdim=True) / 255
        abs_crop = observation[:, 84:94, 18:25:2, 2].reshape(batch_size, 10, 4)
        abs_sensors = abs_crop.sum(dim=1) / 255
        steer_crop = observation[:, 88, 38:58, 1].reshape(batch_size, -1)
        steering = steer_crop.sum(dim=1, keepdim=True)
        gyro_crop = observation[:, 88, 58:86, 0].reshape(batch_size, -1)
        gyroscope = gyro_crop.sum(dim=1, keepdim=True)
        return speed, abs_sensors.reshape(batch_size, 4), steering, gyroscope
