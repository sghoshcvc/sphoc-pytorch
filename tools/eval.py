import numpy as np
import logging
import torch
import tqdm
from evaluation.retrieval import map_from_query_test_feature_matrices
from evaluation.simple_decoder import decode_simple, decode_char_embeddings
from sklearn.preprocessing import LabelEncoder


def evaluate_cnn_batch(cnn, dataset_loader, args, loss_fn,label_encodings=None):
    logger = logging.getLogger('SPHOC-Experiment::test')
    # set the CNN in eval mode
    # fh = logging.FileHandler('log1.txt')
    # logger.addHandler(fh)
    cnn.eval()
    logger.info('Computing net output:')
    qry_ids = np.zeros((len(dataset_loader), args.test_batch_size), dtype=np.int32)
    class_ids = np.zeros((len(dataset_loader), args.test_batch_size), dtype=np.int32)
    embedding_size = dataset_loader.dataset.embedding_size()

    embeddings = np.zeros((len(dataset_loader), args.test_batch_size, embedding_size), dtype=np.float32)
    outputs = np.zeros((len(dataset_loader), args.test_batch_size, embedding_size), dtype=np.float32)
    embeddings_char = np.zeros((len(dataset_loader), args.test_batch_size, 10), dtype=np.float32)
    out_chars_embedding = np.zeros((len(dataset_loader), args.test_batch_size, 370), dtype=np.float32)
    if cnn.length_embedding:
        lengths = np.zeros((len(dataset_loader), args.test_batch_size, 10), dtype=np.float32)
    loss = 0.0
    nb = 0
    for lc, (word_img, embedding, embedding_char, len_embed, class_id, is_query) in enumerate(tqdm.tqdm(dataset_loader)):
        # if sample_idx > 10000:
        #     break
        if args.gpu_id is not None:
            # in one gpu!!
            word_img = word_img.cuda(args.gpu_id[0])
            embedding = embedding.cuda(args.gpu_id[0])
            embedding_char = embedding_char.cuda(args.gpu_id[0])
            len_embed = len_embed.cuda(args.gpu_id[0])
            # word_img, embedding = word_img.cuda(args.gpu_id), embedding.cuda(args.gpu_id)
        word_img = torch.autograd.Variable(word_img)
        embedding = torch.autograd.Variable(embedding)
        embedding_char = torch.autograd.Variable(embedding_char.long())
        len_embed = torch.autograd.Variable(len_embed)

        # st = lc * word_img.shape[0]
        # en = st + word_img.shape[0]
        # print(st, en)
        ''' BCEloss ??? '''
        # output = torch.sigmoid(cnn(word_img))
        retval = cnn(word_img)
        out_keys = list(retval.keys())
        # print(out_keys)
        for key in out_keys:
            if key == 'phoc':
                output = retval[key]
                loss_val = loss_fn[0](output, embedding) * args.test_batch_size
                loss = loss + 0.33 *loss_val.cpu().item()
                output = torch.sigmoid(output)
                nb = nb + 1
                outputs[lc] = output.data.cpu().numpy()
                embeddings[lc] = embedding.data.cpu().numpy()
            if key == 'length_vec':
                out_length = retval[key]
                loss_len = loss_fn[1](out_length, len_embed) * args.test_batch_size
                loss = loss + 0.33 * loss_len.cpu().item()
                est_length = torch.sigmoid(out_length)
                lengths[lc] = est_length.data.cpu().numpy()

            if key == 'char_vec':
                out_char_embedding = retval[key]
                # print(out_char_embedding.shape)
                loss_char = loss_fn[2](out_char_embedding.contiguous().view(-1, 37), embedding_char.view(-1)) * args.test_batch_size
                loss = loss + 0.33 * loss_char.cpu().item()
                out_chars = torch.softmax(out_char_embedding.contiguous().view(-1, 37), dim=1).view(args.test_batch_size, 10, 37)
                out_chars_embedding[lc] = out_chars.view(args.test_batch_size, -1).data.cpu().numpy()
                embeddings_char[lc] = embedding_char.data.cpu().numpy()

        # if len(retval) == 3:
        #     output, out_length, out_char_embedding = retval
        #     loss_val = loss_fn[0](output, embedding) * args.test_batch_size
        #     # same loss function is used to calculate loss for length
        #     loss_len = loss_fn[1](out_length, len_embed) * args.test_batch_size
        #     loss_char = loss_fn[2](out_char_embedding, embedding_char) * args.test_batch_size
        #     # hard coded weighting function between loss
        #     loss = loss +( 0.33 * loss_val.cpu().item() + 0.33 * loss_len.cpu().item() + 0.33 * loss_char.cpu().item())
        #     output = torch.sigmoid(output)
        #
        #     est_length = torch.sigmoid(out_length)
        #     out_chars = torch.softmax(out_char_embedding)
        #
        # elif len(retval) == 2:
        #     output, out_length = retval
        #     loss_val = loss_fn(output, embedding) * args.batch_size
        #     # same loss function is used to calculate loss for length
        #     loss_len = loss_fn(out_length, len_embed) * args.batch_size
        #     # hard coded weighting function between loss
        #     loss = loss + (0.5 * loss_val.cpu().item() + 0.5 * loss_len.cpu().item())
        #     output = torch.sigmoid(output)
        #
        #     est_length = torch.sigmoid(out_length)
        # else:
        #     output = retval
        #     loss_val = loss_fn(output, embedding) * args.batch_size
        #     loss_val_total = loss_val
        #     loss = loss + (loss_val.cpu().item() * output.shape[0])
        #     output = torch.sigmoid(output)


        # if cnn.decoder:
        #
        # if cnn.length_embedding:

        # print(lengths.shape)
        class_ids[lc] = class_id.numpy().flatten()
        # print is_query.shape
        qry_ids[lc] = is_query.byte().numpy().flatten()
        # print qry_ids
        # if is_query[0] == 1:
        # qry_ids.append(sample_idx)  #[sample_idx] = is_query[0]

    '''
    # find queries

    unique_class_ids, counts = np.unique(class_ids, return_counts=True)
    qry_class_ids = unique_class_ids[np.where(counts > 1)[0]]

    # remove stopwords if needed

    qry_ids = [i for i in range(len(class_ids)) if class_ids[i] in qry_class_ids]
    '''
    qry_ids = qry_ids.flatten()
    qry_ids = np.where(qry_ids == 1)[0]
    # print(qry_ids)
    outputs = outputs.reshape((-1, embedding_size))
    out_chars_embedding = out_chars_embedding.reshape((-1, 370))
    if cnn.length_embedding:
        lengths = lengths.reshape((-1, 10))
    class_ids = class_ids.flatten()
    qry_outputs = outputs[qry_ids]
    qry_class_ids = class_ids[qry_ids]
    # lengths = lengths[qry_ids]
    # print(outputs.shape)
    # print(qry_outputs.shape)

    # run word spotting
    logger.info('Computing mAPs...')

    _, ave_precs_qbe = map_from_query_test_feature_matrices(query_features=qry_outputs,
                                                            test_features=outputs,
                                                            query_labels=qry_class_ids,
                                                            test_labels=class_ids,
                                                            metric='cosine',
                                                            drop_first=True)
    # print(ave_precs_qbe)
    mAP = np.mean(ave_precs_qbe[ave_precs_qbe > 0]) * 100
    accuracy = 0.0
    accuracy_char = 0.0
    est_labels = []
    if cnn.decoder:
        est_word_list_char = decode_char_embeddings(out_chars_embedding,lengths,class_ids,label_encodings)
        for word in est_word_list_char:
            try:
                if type(label_encodings) is LabelEncoder:
                    est_labels.append(label_encodings.transform([word]))
                else:
                    est_labels.append(label_encodings.index(word))
            except:
                est_labels.append(-1)
        corrects = np.array(est_labels) == class_ids
        accuracy_char = np.sum(corrects) / len(est_labels)
    if cnn.length_embedding:
        est_word_list = decode_simple(outputs, lengths,class_ids,label_encodings)
        # print(est_word_list[0])
        est_labels = []
        for word in est_word_list:
            try:
                if type(label_encodings) is LabelEncoder: # for IAM
                    est_labels.append(label_encodings.transform([word]))
                else:
                    est_labels.append(label_encodings.index(word)) # for Synth text/svt
            except:
                est_labels.append(-1)
        # est_labels = [label_encodings.transform(s) for s in est_word_list]
        corrects = np.array(est_labels) == class_ids
        accuracy = np.sum(corrects) / len(est_labels)
    loss = loss / np.float(nb)

    # logger.info('mAP: %3.2f %f', mAP, loss)
    # clean up -> set CNN in train mode again
    cnn.train()
    return loss, mAP, accuracy, accuracy_char
