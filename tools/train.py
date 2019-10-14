'''
Created on Sep 7, 2019

@author: Suman Ghosh

- works on GW + IAM + Synth-text dataset
- new way to load dataset
- augmentation with dataloader
- Hardcoded selections (augmentation - default:YES, load pretrained model with hardcoded name...
- do not normalize with respect to iter size (or batch size) for speed
- add fixed size selection (not hardcoded)
- save and load hardcoded name 'SPHOC.pt'
'''
import argparse
import logging
import sys

import numpy as np
import torch.autograd
import torch.cuda
import torch.nn as nn
import torch.optim
# import tqdm
from torch.utils.data import DataLoader

sys.path.insert(0, '/home/sghosh/sphoc-pytorch')
import copy
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from dataset.iam_alt import IAMDataset
from dataset.synth_data import SynthDataSet
from dataset.SVT_data import SVTDataSet

#from cnn_ws.transformations.homography_augmentation import HomographyAugmentation
# from cnn_ws.losses.cosine_loss import CosineLoss

from models.encdec import Autoencoder
from models.sphoc_model import SPHOC
#from torch.utils.data.dataloader import DataLoaderIter
from torch.utils.data.sampler import WeightedRandomSampler
from torch.nn.parameter import Parameter
from tools import eval as test
# from cnn_ws.utils.save_load import my_torch_save, my_torch_load
# from beam_search_decoder import gen_sample
# from simple_decoder import decode_simple


def learning_rate_step_parser(lrs_string):
    return [(int(elem.split(':')[0]), float(elem.split(':')[1])) for elem in lrs_string.split(',')]

def train():
    logger = logging.getLogger('PHOCNet-Experiment::train')
    logger.info('--- Running PHOCNet Training ---')

    # argument parsing
    parser = argparse.ArgumentParser()    
    # - train arguments
    parser.add_argument('--learning_rate_step', '-lrs', type=learning_rate_step_parser, default='30000:1e-4,60000:1e-4,150000:1e-5',
                        help='A dictionary-like string indicating the learning rate for up to the number of iterations. ' +
                             'E.g. the default \'70000:1e-4,80000:1e-5\' means learning rate 1e-4 up to step 70000 and 1e-5 till 80000.')
    parser.add_argument('--momentum', '-mom', action='store', type=float, default=0.9,
                        help='The momentum for SGD training (or beta1 for Adam). Default: 0.9')
    parser.add_argument('--momentum2', '-mom2', action='store', type=float, default=0.999,
                        help='Beta2 if solver is Adam. Default: 0.999')
    parser.add_argument('--delta', action='store', type=float, default=1e-8,
                        help='Epsilon if solver is Adam. Default: 1e-8')
    parser.add_argument('--solver_type', '-st', choices=['SGD', 'Adam'], default='Adam',
                        help='Which solver type to use. Possible: SGD, Adam. Default: Adam')
    parser.add_argument('--display', action='store', type=int, default=500,
                        help='The number of iterations after which to display the loss values. Default: 100')
    parser.add_argument('--test_interval', action='store', type=int, default=2000,
                        help='The number of iterations after which to periodically evaluate the PHOCNet. Default: 500')
    parser.add_argument('--iter_size', '-is', action='store', type=int, default=8,
                        help='The batch size after which the gradient is computed. Default: 10')
    parser.add_argument('--batch_size', '-bs', action='store', type=int, default=1,
                        help='The batch size after which the gradient is computed. Default: 1')
    parser.add_argument('--test_batch_size', '-tbs', action='store', type=int, default=1,
                        help='The batch size after which the gradient is computed. Default: 1')
    parser.add_argument('--weight_decay', '-wd', action='store', type=float, default=0.00005,
                        help='The weight decay for SGD training. Default: 0.00005')
    #parser.add_argument('--gpu_id', '-gpu', action='store', type=int, default=0,
    #                    help='The ID of the GPU to use. If not specified, training is run in CPU mode.')
    parser.add_argument('--gpu_id', '-gpu', action='store',
                        type=lambda str_list: [int(elem) for elem in str_list.split(',')],
                        default='0',
                        help='The ID of the GPU to use. If not specified, training is run in CPU mode.')
    # - experiment arguments
    parser.add_argument('--min_image_width_height', '-miwh', action='store', type=int, default=26,
                        help='The minimum width or height of the images that are being fed to the AttributeCNN. Default: 26')
    parser.add_argument('--phoc_unigram_levels', '-pul',
                        action='store',
                        type=lambda str_list: [int(elem) for elem in str_list.split(',')],
                        default='1,2,3,4,5,6,7,8,9,10',
                        help='The comma seperated list of PHOC unigram levels. Default: 1,2,4,8')
    parser.add_argument('--embedding_type', '-et', action='store',
                        choices=['phoc', 'spoc', 'dctow', 'phoc-ppmi', 'phoc-pruned'],
                        default='phoc',
                        help='The label embedding type to be used. Possible: phoc, spoc, phoc-ppmi, phoc-pruned. Default: phoc')
    parser.add_argument('--fixed_image_size', '-fim', action='store',
                        type=lambda str_tuple: tuple([int(elem) for elem in str_tuple.split(',')]),
                        default=None,
                        help='Specifies the images to be resized to a fixed size when presented to the CNN. Argument must be two comma seperated numbers.')
    parser.add_argument('--dataset', '-ds', required=True, choices=['gw','iam','synth'], default= 'gw',
                        help='The dataset to be trained on')
    parser.add_argument('--decoder', '-dec', required=False, type=bool, default=False,
                        help='Whether decoder is trained or not')
    args = parser.parse_args()

    # sanity checks
    if not torch.cuda.is_available():
        logger.warning('Could not find CUDA environment, using CPU mode')
        args.gpu_id = None

    # print out the used arguments
    logger.info('###########################################')
    logger.info('Experiment Parameters:')
    for key, value in vars(args).items():
        logger.info('%s: %s', str(key), str(value))
    logger.info('###########################################')

    # prepare datset loader
    #TODO: add augmentation
    logger.info('Loading dataset %s...', args.dataset)
    # if args.dataset == 'gw':
    #     train_set = GWDataset(gw_root_dir='../../../pytorch-phocnet/data/gw',
    #                           cv_split_method='almazan',
    #                           cv_split_idx=1,
    #                           image_extension='.tif',
    #                           embedding=args.embedding_type,
    #                           phoc_unigram_levels=args.phoc_unigram_levels,
    #                           fixed_image_size=args.fixed_image_size,
    #                           min_image_width_height=args.min_image_width_height)

    if args.dataset == 'iam':
        train_set = IAMDataset(gw_root_dir='../data/IAM',
                               image_extension='.png',
                               embedding=args.embedding_type,
                               phoc_unigram_levels=args.phoc_unigram_levels,
                               min_image_width_height=args.min_image_width_height,
                               fixed_image_size=None)
        test_set = copy.copy(train_set)

        train_set.mainLoader(partition='train')
        test_set.mainLoader(partition='test', transforms=None)
    if args.dataset == 'synth':
        train_set = SynthDataSet(root_dir='../data/synth/words/', phoc_levels=args.phoc_unigram_levels)
        test_set = SVTDataSet(root_dir='../data/svt/words/', phoc_levels=args.phoc_unigram_levels)
        label_encoder = test_set.get_label()

    


    # augmentation using data sampler
    n_train_images = 500000
    augmentation = False

    if augmentation:
        train_loader = DataLoader(train_set,
                                  sampler=WeightedRandomSampler(train_set.weights, n_train_images),
                                  batch_size=args.batch_size,
                                  num_workers=8)
    else:
        train_loader = DataLoader(train_set,
                                  batch_size=args.batch_size, shuffle=True,
                                  num_workers=8)

    train_loader_iter = iter(train_loader)
    test_loader = DataLoader(test_set,
                             batch_size=args.test_batch_size,
                             shuffle=False,
                             num_workers=8)
    # load CNN
    logger.info('Preparing PHOCNet...')

    #cnn = PHOCNet(n_out=train_set[0][1].shape[0],
    #              input_channels=1,
    #              gpp_type='gpp',
    #              pooling_levels=([1], [5]))
    if args.decoder:
        sphoc_model = SPHOC(enocder_type='phocnet', decoder=True)
    else:
        sphoc_model = SPHOC(enocder_type='phocnet', length_embedding=True, position_embedding=True)


    ## pre-trained!!!!
    load_pretrained = False
    if load_pretrained:
        #cnn.load_state_dict(torch.load('PHOCNet.pt', map_location=lambda storage, loc: storage))
        logger.info('Preparing PHOCNet... loading pretrained model')
        sphoc_model = torch.load('SPHOC_best.pt')

        logger.info('Preparing PHOCNet... loaded pretrained model')

    else:

        sphoc_model.init_weights()

    phoc_loss = nn.BCEWithLogitsLoss(reduction='mean')
    if args.decoder:
        char_loss = nn.CrossEntropyLoss(reduction='mean')
        # recon_loss = nn.MSELoss(reduction='mean')


    # move CNN to GPU
    if args.gpu_id is not None:
        if len(args.gpu_id) > 1:
            sphoc_model = nn.DataParallel(sphoc_model, device_ids=args.gpu_id)
            sphoc_model.cuda()
        else:
            sphoc_model.cuda(args.gpu_id[0])

    # run training
    lr_cnt = 0
    max_iters = args.learning_rate_step[-1][0]
    if args.solver_type == 'SGD':

        optimizer = torch.optim.SGD([
            {'params': sphoc_model.encoder.parameters()},
            {'params': sphoc_model.att.parameters(), 'lr': 1e-3}
        ],
        args.learning_rate_step[0][1],
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    if args.solver_type == 'Adam':
        optimizer = torch.optim.Adam([
            {'params': sphoc_model.encoder.parameters()},
            {'params': sphoc_model.att.parameters()},
            {'params': sphoc_model.length_estimator.parameters()},
            {'params': sphoc_model.fc2.parameters()}
        ], args.learning_rate_step[0][1], weight_decay=args.weight_decay)
    optimizer.zero_grad()
    logger.info('Training:')
    best_map = 0
    fmap = open('map.txt', 'w')
    
    for iter_idx in range(max_iters):
        if iter_idx % args.test_interval == 0: # and iter_idx > 0:
            logger.info('Evaluating net after %d iterations', iter_idx)
            loss_test, mAp, acc = test.evaluate_cnn_batch(cnn=sphoc_model,
                               dataset_loader=test_loader,
                               args=args, loss_fn=phoc_loss,label_encodings=label_encoder)
            logger.info('mAP after %d iterations: %3.2f %f %f', iter_idx, mAp, acc, loss_test)
            fmap=open('map_sphoc.txt','a')
            fmap.write('mAP after '+ str(iter_idx)+' iterations:'+str(mAp)+', loss: '+str(loss_test)+'\n')
            fmap.close()
            if mAp > best_map:
                best_map = mAp
                torch.save(sphoc_model, 'SPHOC_best.pt')
        # if args.gpu_id is not None:
        #     if len(args.gpu_id) > 1:
        #         sphoc_model = nn.DataParallel(sphoc_model, device_ids=args.gpu_id)
        #         sphoc_model.cuda()
        #     else:
        #         cnn.cuda(args.gpu_id[0])
        
        for _ in range(args.iter_size):
            if train_loader_iter.batches_outstanding == 0:
                train_loader = DataLoader(train_set,
                                  sampler=WeightedRandomSampler(train_set.weights, n_train_images),
                                  batch_size=args.batch_size,
                                  num_workers=8)
                train_loader_iter = iter(train_loader)
                logger.info('Resetting data loader')
            try: 
                word_img, embedding, len_embed, _, _  = next(train_loader_iter)
                #print('success')
            except StopIteration:
                print('stop iteration')
                pass
            except Exception as e:
                logger.info('Failed to upload to ftp: '+ str(e))

            if args.gpu_id is not None:
                if len(args.gpu_id) > 1:
                    word_img = word_img.cuda()
                    embedding = embedding.cuda()
                    len_embed = len_embed.cuda()
                else:
                    word_img = word_img.cuda(args.gpu_id[0])
                    embedding = embedding.cuda(args.gpu_id[0])
                    len_embed = len_embed.cuda(args.gpu_id[0])
            # print('before cnn fwd')
            word_img = torch.autograd.Variable(word_img)
            embedding = torch.autograd.Variable(embedding)
            len_embed = torch.autograd.Variable(len_embed)
            retval = sphoc_model(word_img)
            if len(retval) == 3:
                output, out_length, pred_chars = retval
                loss_val = phoc_loss(output, embedding) * args.batch_size
                # same loss function is used to calculate loss for length
                loss_len = phoc_loss(out_length, len_embed) * args.batch_size
                # loss_char = char_loss(pred_chars,)
                # hard coded weighting function between loss
                loss_val_total = 0.8 * loss_val + 0.2 * loss_len
            elif len(retval) == 2:
                output, out_length = retval
                loss_val = phoc_loss(output, embedding) * args.batch_size
                # same loss function is used to calculate loss for length
                loss_len = phoc_loss(out_length, len_embed) * args.batch_size
                # hard coded weighting function between loss
                loss_val_total = 0.5 * loss_val + 0.5 * loss_len
            else:
                output = retval
                loss_val = phoc_loss(output, embedding) * args.batch_size
                loss_val_total = loss_val

            ''' BCEloss ??? '''
            # print(output.shape)
            # print(out_length.shape)

            loss_val_total.backward()
            # print('before cnn bwd')
        optimizer.step()
        optimizer.zero_grad()

        # mean runing errors??
        if (iter_idx+1) % args.display == 0:
            if sphoc_model.length_embedding:
                logger.info('Iteration %*d: %f %f', len(str(max_iters)), iter_idx+1, loss_val.item(), loss_len.item())
            else:
                logger.info('Iteration %*d: %f', len(str(max_iters)), iter_idx + 1, loss_val.item())


        # change lr
        if (iter_idx + 1) == args.learning_rate_step[lr_cnt][0] and (iter_idx+1) != max_iters:
            lr_cnt += 1
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.learning_rate_step[lr_cnt][1]
        
        

        #if (iter_idx + 1) % 10000 == 0:
        #    torch.save(cnn.state_dict(), 'PHOCNet.pt')
            # .. to load your previously training model:
            #cnn.load_state_dict(torch.load('PHOCNet.pt'))

    #torch.save(cnn.state_dict(), 'PHOCNet.pt')
    torch.save(sphoc_model, 'sphoc_PHOCNet.pt')
    #fmap.close()



if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s, %(levelname)s, %(name)s] %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    train()
