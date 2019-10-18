import torch
import torch.nn as nn
import torch.nn.functional as F
from models.textinception import TextInception3
from models.Length_estimator import Length_estimator
from models.myphocnet import PHOCNet
from models.decoder import PhocDecoder


class SPHOC(nn.Module):
    '''
    Network class for generating SPHOC architecture
    '''

    def __init__(self, enocder_type='inception', length_embedding=False, position_embedding=False, decoder=False, max_len=10, voc_size=36):
        super(SPHOC, self).__init__()
        # some sanity checks
        batchNorm_momentum = 0.1
        self.len = max_len
        self.decoder = decoder
        self.length_embedding = length_embedding
        self.pos_embedding = position_embedding

        if enocder_type=='inception':
            self.encoder = TextInception3()
            self.nChannels = 2048
            self.pos_len = 11
        elif enocder_type == 'resnet':
            self.encoder = TextResnet()
            self.nChannels = 2048
            self.pos_len = 11
        else:
            self.encoder = PHOCNet()
            self.nChannels = 512
            self.pos_len = 26
            self.H =8
            self.W =18
        # self.encoder = PHOCNet(n_out=1980)
        self.length_estimator = Length_estimator(filter_size=self.nChannels, ht=self.H, wd=self.W, max_len=max_len)
        if length_embedding and position_embedding:

            nChannels = self.nChannels + max_len + self.pos_len
        elif position_embedding:
            # elf.length_embedding = True
            # self.pos_embedding = True
            nChannels = self.nChannels + self.pos_len
        elif length_embedding:
            # self.length_embedding = True
            # self.pos_embedding = True
            nChannels = self.nChannels + max_len
        else: # without any position or length info
            nChannels = self.nChannels
            # self.length_embedding = False
            # self.pos_embedding = False


        print(nChannels)
        nChannels = self.nChannels
        self.att = nn.Conv2d(in_channels=nChannels, out_channels=55, kernel_size=1, stride=1)
        #self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(self.nChannels, voc_size)
        if decoder:
            self.decoder = True
            self.decoder = PhocDecoder(self.nChannels, maxlen=max_len, voc_size=voc_size)



    def length_embed(self, filters, length_vec):

        batch_size, _, ht, wd = filters.shape
        # print(filters.shape)
        length_vec = length_vec.unsqueeze(-1)
        # print('length_emb')
        # print(length_vec.shape)
        length_vec = length_vec.expand(-1, -1, ht * wd)
        # print(length_vec.shape)
        length_vec = length_vec.view(batch_size, -1, ht, wd)
        # print(length_vec.shape)
        # length_vec = length_vec.unsqueeze(0).expand(batch_size, -1, -1, -1)
        filters = torch.cat((filters, length_vec), dim=1)
        return filters

    def position_embed(self, y_filters):

        batch_size, _, ht, wd = y_filters.shape
        pos_vec = torch.zeros(ht, wd, ht+wd).float().cuda()
        for i in range(ht):
            for j in range(wd):
                pos_vec[i, j, i] = 1
                pos_vec[i, j, ht+j] = 1
        pos_vec = pos_vec.permute(2, 0, 1)
        pos_vec = pos_vec.unsqueeze(0).expand(batch_size, -1, -1, -1)

        filters = torch.cat((y_filters, pos_vec), dim=1)
        # filters = torch.cat((filters, length_vec), dim=1)
        return filters

    def forward(self, x):

        conv_filters = self.encoder(x)
        # print(conv_filters.shape)
        # print('before length estimate')

        # print('After length estimate')

        # y = F.relu(self.conv_1x1(x))
        #print(y_filters.shape)
        batch_size, feat_size, ht, wd = conv_filters.shape
        # filters_pos = y_filters  # concat position and length for each
        # x_pos = torch.arange(wd)
        # y_pos = torch.arange(ht)
        filters_emb = conv_filters
        if self.pos_embedding:
            filters_emb = self.position_embed(conv_filters)
        if self.length_embedding:
            len_vec, length_vec_sigmoid = self.length_estimator(conv_filters)
            filters_emb = self.length_embed(filters_emb, length_vec_sigmoid)
        # print(filters_pos_len.shape)
        part_attention = F.relu(self.att(conv_filters))  # batch_size x sub_hocs x H x W
        #
        # # y_attention = y_attention.view(batch_size*55, 1, y_attention.shape[2], y_attention.shape[3])
        # # y_attention = y_attention.expand(-1, 512, -1, -1)
        # # y = y.repeat_interleave(55, dim=0)
        # # att_y = y * y_attention
        # # att_y_emb = torch.sum(att_y.view(batch_size*55, 512, -1), dim=2)
        y = conv_filters.view(batch_size, feat_size, -1)
        part_attention = part_attention.view(batch_size, 55, -1)
        part_attention = part_attention.transpose(1, 2)
        att_y = torch.bmm(y, part_attention)
        att_y = att_y.transpose(1, 2)
        att_y = att_y.contiguous().view(-1, feat_size)
        #att_y = self.fc1(att_y)
        #att_y = F.dropout(att_y, p=0.5, training=self.training)
        phoc = self.fc2(att_y).view(batch_size, -1)
        retval['phoc'] =phoc
        if self.decoder:
            pred_chars = self.decoder(att_y.view(batch_size, -1, 512))
            retval['char_vec'] = pred_chars

        #phoc = y_filters

        #y = self.decoder(y_filters,indices1,indices2,size1, size2)
        if self.length_embedding:
            retval['length_vec'] = len_vec
        

        return retval

    def init_weights(self):
        self.apply(SPHOC._init_weights_he)

    '''
    @staticmethod
    def _init_weights_he(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            #nn.init.kaiming_normal(m.weight.data)
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, (2. / n)**(1/2.0))
            if hasattr(m, 'bias'):
                nn.init.constant(m.bias.data, 0)
    '''

    @staticmethod
    def _init_weights_he(m):
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, (2. / n) ** (1 / 2.0))
        if isinstance(m, nn.Linear):
            n = m.out_features
            m.weight.data.normal_(0, (2. / n) ** (1 / 2.0))
            # nn.init.kaiming_normal(m.weight.data)
            nn.init.constant(m.bias.data, 0)
