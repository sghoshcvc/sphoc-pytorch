# from cnn_ws.models.resnet import resnet152
# from cnn_ws.models.myphocnet import PHOCNet_classic
import torch
import torch.nn as nn
import torch.nn.functional as F
#from cnn_ws.models.resnet import resnet152
from models.textinception import TextInception3
from models.myphocnet import PHOCNet
from models.textinception import TextInception3
#from cnn_ws.models.myecoder import Decoder


class Autoencoder(nn.Module):
    '''
    Network class for generating PHOCNet and TPP-PHOCNet architectures
    '''

    def __init__(self, input_channels=1):
        super(Autoencoder, self).__init__()
        # some sanity checks
        batchNorm_momentum = 0.1

        self.encoder = TextInception3()
        # self.encoder = PHOCNet(n_out=1980)
        self.length_estimator = Length_estimator(max_len=10)
        self.att = nn.Conv2d(in_channels=548, out_channels=55, kernel_size=1, stride=1)
        #self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 36)

        #self.decoder = Decoder()

    def length_embed(self, filters, length_vec):

        batch_size, _, ht, wd = filters.shape
        print(filters.shape)
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
        pos_vec = torch.zeros(8, 18, 26).float().cuda()
        for i in range(ht):
            for j in range(wd):

                    pos_vec[i, j, i] = 1
                    pos_vec[i, j, ht+j] = 1

        # x_pos = torch.arange(wd).float().cuda()
        # y_pos = torch.arange(ht).float().cuda()
        # x_pos = x_pos.view(1, wd)
        # y_pos = y_pos.view(ht, 1)
        # y_pos = y_pos.repeat(1, wd)
        # x_pos = x_pos.repeat(ht, 1)
        # pos_vec = torch.cat((x_pos.unsqueeze(0), y_pos.unsqueeze(0)), dim=0)
        # print(pos_vec.shape)
        # print(pos_vec)
        # print(pos_vec[0, 0, :])
        pos_vec = pos_vec.permute(2, 0, 1)
        # print(pos_vec.shape)
        pos_vec = pos_vec.unsqueeze(0).expand(batch_size, -1, -1, -1)
        # print(pos_vec.shape)
        # print(length_vec.shape)
        # length_vec = length_vec.unsqueeze(-1)
        # print(length_vec.shape)
        # length_vec = length_vec.expand(-1, -1, ht * wd)
        # print(length_vec.shape)
        # length_vec = length_vec.view(batch_size, -1, ht, wd)
        # print(length_vec.shape)
        # length_vec = length_vec.unsqueeze(0).expand(batch_size, -1, -1, -1)
        # print(length_vec.shape)

        filters = torch.cat((y_filters, pos_vec), dim=1)
        # filters = torch.cat((filters, length_vec), dim=1)
        return filters

    def forward(self, x):

        conv_filters = self.encoder(x)
        print(conv_filters.shape)
        # print('before length estimate')

        # print('After length estimate')

        # y = F.relu(self.conv_1x1(x))
        #print(y_filters.shape)
        batch_size, feat_size, ht, wd = conv_filters.shape
        # filters_pos = y_filters  # concat position and length for each
        # x_pos = torch.arange(wd)
        # y_pos = torch.arange(ht)
        filters_pos = self.position_embed(conv_filters)
        len_vec, length_vec_sigmoid = self.length_estimator(conv_filters)
        filters_pos_len = self.length_embed(filters_pos, length_vec_sigmoid)
        # print(filters_pos_len.shape)
        part_attention = F.relu(self.att(filters_pos_len))  # batch_size x sub_hocs x H x W
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
        #phoc = y_filters

        #y = self.decoder(y_filters,indices1,indices2,size1, size2)
        return phoc, len_vec
       #
       #
       #
       #
       #  # print(x.shape)
       #  batch_size = x.shape[0]
       #  # y = F.relu(self.bn11(self.conv1_1(x)))
       #  # # print(y.shape)
       #  # # y = F.relu(self.conv1_2(y))
       #  # size1 = y.size()
       #  # y, indices1 = F.max_pool2d(y, kernel_size=2, stride=2, padding=0, return_indices=True)
       #  # # print(y.shape)
       #  # y = F.relu(self.bn21(self.conv2_1(y)))
       #  # # print(y.shape)
       #  # y = F.relu(self.conv2_2(y))
       #  # # print(y.shape)
       #  # size2 = y.size()
       #  # y, indices2 = F.max_pool2d(y, kernel_size=2, stride=2, padding=0, return_indices=True)
       #  # # print(y.shape)
       #  # y = F.relu(self.conv3_1(y))
       #  # # print(y.shape)
       #  # y = F.relu(self.bn32(self.conv3_2(y)))
       #  # # print(y.shape)
       #  # # y = F.relu(self.conv3_3(y))
       #  # # y = F.relu(self.conv3_4(y))
       #  # # y = F.relu(self.conv3_5(y))
       #  # # y = F.relu(self.conv3_6(y))
       #  # y = F.relu(self.conv4_1(y))
       #  # # print(y.shape)
       #  # y = F.relu(self.bn42(self.conv4_2(y)))
       #  # y = F.relu(self.model_ft(x))
       #  # print(y.shape)
       #
       #  # rec_img = self.defc1(att_y)
       #  # y_img = F.dropout(rec_img.view(batch_size, feat_size, 3, 8), training=self.training)
       #
       #  # print(phoc[0])
       #  #reconstruction
       #  # y_img = F.relu(self.bn42_d(self.deconv4_2(att_y)))
       #  # # print(y.shape)
       #  # y_img = F.relu(self.deconv4_1(y_img))
       #  # # print(y.shape)
       #  # y_img = F.relu(self.bn32_d(self.deconv3_2(y_img)))
       #  # # print(y.shape)
       #  # y_img = F.relu(self.deconv3_1(y_img))
       #  # # print(y.shape)
       #  # y_img = self.unpool1(y_img, indices2.repeat_interleave(55, dim=0), torch.Size((batch_size*55, 128, 35, 75)))
       #  # # y_img = self.unpool1(y_img, indices2, size2)
       #  # # print(y.shape)
       #  # y_img = F.relu(self.deconv2_2(y_img))
       #  # # print(y.shape)
       #  # y_img = F.relu(self.bn21_d(self.deconv2_1(y_img)))
       #  # # print(y.shape)
       #  # y_img = self.unpool1(y_img, indices1.repeat_interleave(55, dim=0), torch.Size((batch_size*55, 64, 70, 150)))
       #  #
       #  # # y_img = self.unpool2(y_img, indices1, size1)
       #  # # print(y_img.shape)
       #  # y_img = F.relu(self.deconv1_1(y_img)) # 55*batch_size x 1 x H x W
       #  # y_img = y_img.view(batch_size,-1, y_img.shape[2], y_img.shape[3])
       #  # img, _ = torch.max(y_img, dim=1, keepdim=True)
       #
       #  #
       #  # phoc = torch.zeros(batch_size, 1980).cuda()
       #  # img = torch.zeros(batch_size, 55, 70, 150).cuda()
       #  #
       #  #
       #  # for i in range(y_attention.shape[1]): # number of attentions
       #  #     y_attention_hoc = y_attention[:, i, :, :] # batch_size x 1 x H x W
       #  #     # print (y_attention_hoc.shape)
       #  #     y_attention_hoc = y_attention_hoc.unsqueeze(1).expand(-1, 512, -1, -1)
       #  #     att_y = y * y_attention_hoc # batch_size x 512 x H x W
       #  #     att_y_emb = torch.sum(att_y.view(batch_size, 512, -1), dim=2) # batch_size x 512
       #  #     # print(att_y_emb.size())
       #  #     st= (i) * 36
       #  #     en = (i+1) * 36
       #  #     phoc[:, st:en] = self.fc(att_y_emb) # batch_size x 36
       #  #     # reconstruction
       #  #     y_img = F.relu(self.bn42_d(self.deconv4_2(att_y)))
       #  #     # print(y.shape)
       #  #     y_img = F.relu(self.deconv4_1(y_img))
       #  #     # print(y.shape)
       #  #     y_img = F.relu(self.bn32_d(self.deconv3_2(y_img)))
       #  #     # print(y.shape)
       #  #     y_img = F.relu(self.deconv3_1(y_img))
       #  #     # print(y.shape)
       #  #     # y = self.unpool1(y, indices2.expand(55, -1, -1, -1), torch.Size((55, 128, 35, 75)))
       #  #     y_img = self.unpool1(y_img, indices2, size2)
       #  #     # print(y.shape)
       #  #     y_img = F.relu(self.deconv2_2(y_img))
       #  #     # print(y.shape)
       #  #     y_img = F.relu(self.bn21_d(self.deconv2_1(y_img)))
       #  #     # print(y.shape)
       #  #     y_img = self.unpool2(y_img, indices1, size1)
       #  #     # print(y_img.shape)
       #  #     y_img = F.relu(self.deconv1_1(y_img)) # batch_size x 1 x H x W
       #  #     # print(y_img.shape)
       #  #     img[:, i, :, :] = y_img.squeeze()
       #  # img, _ = torch.max(img, dim=1, keepdim=True)
       #  #
       #  #
       #
       #
       #  # print(y_attention.shape)
       #  # y_attention = y_attention.permute(1, 0, 2, 3)
       #  # print(y_attention.shape)
       #  # y_attention = y_attention.expand(-1, 512, -1, -1)
       #  # print(y_attention.shape)
       #  # y_attention.shape
       #  # att_y = y.expand(55, -1, -1, -1) * y_attention
       #  # print(y.shape)
       #  # phoc = self.fc(torch.sum(att_y.view(55, 512, -1), dim=2))
       #  # phoc = phoc.view(-1).unsqueeze(0)
       #  # print(att_y.shape)
       #
       #  # y, _ = torch.max(y, dim=0, keepdim=True)
       #  # print(y.shape)
       # # y = F.relu(self.conv4_3(y))
       #  return phoc

    def init_weights(self):
        self.apply(PHOCNet._init_weights_he)


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
            #nn.init.kaiming_normal(m.weight.data)
            nn.init.constant(m.bias.data, 0)
