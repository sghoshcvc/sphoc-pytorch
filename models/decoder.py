import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil


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
        self.att_subhoc = nn.Conv1d(in_channels=filters_channels, out_channels=maxlen, kernel_size=1, stride=1)
        self.char_fc = nn.Linear(filters_channels, voc_size)

    def position_embed(self, y_filters, maxlen=10):

        batch_size, _, num_parts = y_filters.shape
        pos_vec = torch.zeros(num_parts, 2*maxlen).float().cuda()
        for i in range(num_parts):
            for j in range(maxlen):
                pos_vec[i, j] = 1
                pos_vec[i, j] = 1
        pos_vec = pos_vec.permute(2, 0, 1)
        pos_vec = pos_vec.unsqueeze(0).expand(batch_size, -1, -1, -1)

        filters = torch.cat((y_filters, pos_vec), dim=1)
        # filters = torch.cat((filters, length_vec), dim=1)
        return filters

    def forward(self, filters, maxlen=10):

        batch_size = filters.shape[0]
        filters_pos = self.position_embed(filters)

        part_attention = F.relu(self.att_subhoc(filters_pos))
        # y = filters.view(batch_size,feat_size, -1)
        part_attention = part_attention.view(batch_size, maxlen, -1)
        part_attention = part_attention.transpose(1, 2)
        att_y = torch.bmm(filters, part_attention)
        att_y = att_y.transpose(1, 2)
        att_y = att_y.contiguous().view(-1, maxlen)
        # att_y = self.fc1(att_y)
        # att_y = F.dropout(att_y, p=0.5, training=self.training)
        pred_chars = self.char_fc(att_y).view(batch_size, -1)

        return pred_chars


