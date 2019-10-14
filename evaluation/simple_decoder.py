import numpy as np


def decode_simple(output, lengths,class_ids,label_encodings, alpha=0.0, trie=None):
    unigrams = [chr(i) for i in list(range(ord('a'), ord('z') + 1)) + list(range(ord('0'), ord('9') + 1))]
    print(output.shape)
    print(lengths.shape)
    words = []
    f = open('word_out.txt', 'w')
    for cnt in range(output.shape[0]):
        # print(est_length)
        # est_length = np.where(lengths[cnt] > 0.5)[0]
        # est_length = est_length.data.cpu().numpy()
        # print(est_length)
        est_length = lengths[cnt].flatten() > 0.5
        if np.any(est_length):
            est_length = np.max(np.where(est_length))+1
        else:
            est_length =5
        # print(est_length)
        nstart = int(((est_length-1)*est_length)/2)
        nend = int(nstart+est_length)
        decoder_in = output[cnt].reshape((-1, len(unigrams)))
        # print(nstart)
        # print(nend)
        decoder_in = decoder_in[nstart:nend]

        maxval = []

        # print(decoder_in.shape)
        for i in range(decoder_in.shape[0]):
            # x=np.argmax(decoder_in[i])
            # print(x)
            maxval.append(np.argmax(decoder_in[i]))
        # print(maxval)
        word = [unigrams[i] for i in maxval]
        word = ''.join(word)
        
    #for word in words:
        f.write(word.upper()+'\t'+ label_encodings[class_ids[cnt]]+'\n')
        words.append(word.upper())
        #gt.append(label_encodings[class_ids[cnt]]
    f.close()
    return  words