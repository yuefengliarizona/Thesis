import torch
from Mask1 import CustomizedLinear
import torch.nn as nn
import torch.nn.functional as F

class SimpleModel(torch.nn.Module):
    def __init__(self,Mask):
        super(SimpleModel, self).__init__()
        self.Mask=Mask

        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        #self.dropout = nn.Dropout(0.25)


        #self.linear1=torch.nn.Sequential(
        #CustomizedLinear(Mask[0], bias=None),  # dimmentions is set from mask.size
        self.fc1 =torch.nn.Linear(16*4*4,Mask[0].shape[0], bias=None)
        #torch.nn.ReLU(),
        self.layers = nn.ModuleList()
        for i in range(len(Mask)):
            self.layers.append(CustomizedLinear(Mask[i], bias=None))
        #torch.nn.ReLU(),
        self.fc5 = torch.nn.Linear(Mask[-1].shape[1], 10, bias=None)

        #)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16* 4 * 4)
        x = F.relu(self.fc1(x))
        #x= self.dropout(x)
        for layer in self.layers:
            x = F.relu(layer(x))
            #x = self.dropout(x)
        x = self.fc5(x)
        #x = self.linear1(x)
        #x = self.fc4(x)
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
"""
        return x

