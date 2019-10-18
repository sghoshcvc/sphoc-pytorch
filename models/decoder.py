import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil
import numpy as np

# from cnn_ws.spatial_pyramid_layers.gpp import GPP


class PhocDecoder(nn.Module):
    '''
    Network class for decoding phoc
    input phoc (batch_size X num_level X channels)
    '''

    def __init__(self, filters_channels, maxlen=10, voc_size=36):
        super(PhocDecoder, self).__init__()
        # some sanity checks
        # nChannels = filters.shape[2]
        self.att_subhoc = nn.Conv1d(in_channels=filters_channels+1, out_channels=maxlen, kernel_size=1, stride=1)
        self.char_fc = nn.Linear(filters_channels, voc_size+1)
        num_parts = np.sum(np.arange(1, maxlen+1))
        self.pos_vec = torch.zeros(num_parts).float().cuda()
        cnt = 0
        for i in range(maxlen):
            for j in range(i):
                self.pos_vec[cnt] = 1.0/i
                cnt = cnt+1
        self.pos_vec = self.pos_vec.view(num_parts, -1)


    def position_embed(self, y_filters, maxlen=10):

        batch_size, num_parts, hoc = y_filters.shape
        # print(y_filters.shape)
        # pos_vec = torch.zeros(num_parts).float().cuda()

                # pos_vec[i, j] = 1
        # print(self.pos_vec.shape)
        pos_vec = self.pos_vec.unsqueeze(0).expand(batch_size, -1, -1)
        # print(pos_vec.shape)

        filters = torch.cat((y_filters, pos_vec), dim=2)
        # print(filters.shape)
        # filters = torch.cat((filters, length_vec), dim=1)
        return filters

    def forward(self, filters, maxlen=10):

        batch_size = filters.shape[0]
        # print(filters.shape)
        filters_pos = self.position_embed(filters)
        # print(filters_pos.shape)
        part_attention = F.relu(self.att_subhoc(filters_pos.transpose(1, 2)))
        # print(part_attention.shape)
        # y = filters.view(batch_size,feat_size, -1)
        part_attention = part_attention.view(batch_size, maxlen, -1)
        part_attention = part_attention.transpose(1, 2)
        # print(filters.shape)
        # print(part_attention.shape)
        att_y = torch.bmm(filters.transpose(1, 2), part_attention)
        # print(att_y.shape)
        att_y = att_y.transpose(1, 2)
        #att_y = att_y.contiguous().view(-1, maxlen)
        #print(att_y.shape)
        # att_y = self.fc1(att_y)
        # att_y = F.dropout(att_y, p=0.5, training=self.training)
        pred_chars = self.char_fc(att_y).view(batch_size, -1)

        return pred_chars


