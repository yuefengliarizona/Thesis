import torch
import torch.nn as nn
import torch.nn.functional as F


class T3Model(torch.nn.Module):
    def __init__(self):
        super(T3Model, self).__init__()
        """Dim_HIDDEN1=Mask[0].shape[0]
        Dim_OUTPUT1=Mask[0].shape[1]
        Dim_HIDDEN2=Dim_OUTPUT1
        Dim_OUTPUT2=Mask[1].shape[1]
        Dim_HIDDEN3=Dim_OUTPUT2
        Dim_OUTPUT3=Mask[2].shape[1]
        """
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        """
        self.fc1 = CustomizedLinear(Mask[0], bias=None)
        self.fc2 = CustomizedLinear(Mask[1], bias=None)
        self.fc3 = nn.Linear(16, 10)
        """
        # self.linear1=torch.nn.Sequential(
        # CustomizedLinear(Mask[0], bias=None),  # dimmentions is set from mask.size
        self.fc1 = torch.nn.Linear(16 * 4 * 4, 120, bias=None)
        # torch.nn.ReLU(),
        self.fc2 = torch.nn.Linear(120, 84, bias=None)
        # torch.nn.ReLU(),
        self.fc3 = torch.nn.Linear(84, 50, bias=None)  # dimmentions is set from mask.size
        # torch.nn.ReLU(),
        self.fc4 = torch.nn.Linear(50, 10, bias=None)  # dimmentions is set from mask.size

        # )

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        # x = self.linear1(x)
        # x = self.fc4(x)
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        """
        return x

