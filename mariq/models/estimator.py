import torch
import torch.nn as nn
import numpy as np

class Estimator(nn.Module):
    def __init__(self, input_shape, num_actions):
            super(Estimator, self).__init__()

            self.conv_1 = nn.Sequential(
                nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
                nn.BatchNorm2d(32),
                nn.ReLU()
            )
            self.conv_2 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.BatchNorm2d(64),
                nn.ReLU()
            )
            self.conv_3 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.BatchNorm2d(64),
                nn.ReLU()
            )

            self.fc_1 = nn.Linear(7*7*64, 512)
            self.output = nn.Linear(512, num_actions)


    def forward(self, x):
        x = x.float()
        y = self.conv_1(x)
        # print('conv_1=', y.shape)
        y = self.conv_2(y)
        # print('conv_2=', y.shape)
        y = self.conv_3(y)
        # print('conv_3=', y.shape)
        y = y.reshape(y.size(0), -1)
        # print(y.shape)
        y = self.fc_1(y)
        y = self.output(y)
        return y


