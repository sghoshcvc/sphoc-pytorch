import torch
import torch.nn as nn
import torch.nn.functional as F


class Length_estimator(nn.Module):
    '''
    Network class for generating PHOCNet and TPP-PHOCNet architectures
    '''

    def __init__(self, filter_size=2048, ht=3, wd=8, max_len=10):
        super(Length_estimator, self).__init__()
        # some sanity checks
        # self.pooling_layer_fn = GPP(gpp_type=gpp_type, levels=pooling_levels, pool_type=pool_type)

        # pooling_output_size = self.pooling_layer_fn.pooling_output_size
        self.conv_len = nn.Conv2d(in_channels=filter_size, out_channels=128, kernel_size=1, stride=1)
        # self.fc0 = nn.Linear(512, 8)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, max_len)
        self.lenfc = nn.Sigmoid()

    def forward(self, x):
        # print('length')
        # print(x.shape)
        y = F.relu(self.conv_len(x))
        # print(y.shape)
        # y = self.pooling_layer_fn(x)
        adaptiveAvgPoolHeight = x.shape[2]
        adaptiveAvgPoolWidth = x.shape[3]
        # y = F.relu(self.fc0(x))

        y = F.avg_pool2d(y, kernel_size=(adaptiveAvgPoolHeight,adaptiveAvgPoolWidth))
        y = y.view(y.size(0), -1)
        # print(y.shape)
        y = F.relu(self.fc1(y))
        z = F.dropout(y, p=0.5, training=self.training)
        len_vec = self.fc2(z)
        len_vec_sigmoid = self.lenfc(len_vec)
        # y = F.dropout(y, p=0.5, training=self.training)
        # y = self.fc7(y)
        return len_vec, len_vec_sigmoid
