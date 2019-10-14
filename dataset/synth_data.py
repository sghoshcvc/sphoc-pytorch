import cv2
import numpy as np
from dataset.phoc import build_phoc_descriptor, get_most_common_n_grams
import torch

class SynthDataSet:
    def __init__(self, root_dir, phoc_levels='1,2,3,4,5', dset='train', phoc=None, fixed_image_size=(96,32), randomize=True):
        self._root_dir = root_dir

        if dset == 'train':
            self._data_set = open(self._root_dir + 'annotation_train.txt', 'r').readlines()
        else:
            self._data_set = open(self._root_dir + 'annotation_test.txt', 'r').readlines()

        lex_file = open(self._root_dir + 'lexicon.txt', 'r')
        temp = lex_file.readlines()
        lex_file.close()
        self._lex = [l.strip('\t\n\r').lower() for l in temp]

        self._fixed_image_size = fixed_image_size

        if randomize and dset == 'train':
            # if necessary generate randomize data set
            random_index = np.random.permutation(range(len(self._data_set)))
            self._data_set = [self._data_set[i] for i in random_index]
        if dset == 'test':
            random_index = np.random.permutation(range(len(self._data_set)))
            self._data_set = [self._data_set[i] for i in random_index[:1000]]

        if phoc is None:

            # print self._lex
            unigrams = [chr(i) for i in list(range(ord('a'), ord('z') + 1)) + list(range(ord('0'), ord('9') + 1))]
            # unigrams = get_unigrams_from_strings(word_strings=[elem[1] for elem in words])
            # if use_bigrams:
            #     bigram_levels = [2]
            #     bigrams = get_most_common_n_grams(word_strings)
            # else:
            bigram_levels = None
            bigrams = None
            self._phoc = build_phoc_descriptor(words=self._lex,
                                               phoc_unigrams=unigrams,
                                               bigram_levels=bigram_levels,
                                               phoc_bigrams=bigrams,
                                               unigram_levels=phoc_levels)
        else:
            self._phoc = phoc
        self._phoc = self._phoc.astype(np.float32)

        self._len = [len(w) for w in self._lex]
        self.length_embeddings = np.zeros((len(self._lex), 10),dtype=np.float32)
        for ind, x in enumerate(self._len):
            
            # print(word)
            self.length_embeddings[ind][:x] = 1
            # print(self.length_embeddings[ind])

    def __getitem__(self, index):

        im_file, label = self._data_set[index].split()
        # read image
        # print im_file
        #print(label)
        #print(int(label))
        
        img_orig = cv2.imread(self._root_dir+im_file)

        if img_orig is None:
            im_file, label = self._data_set[0].split()
            img_orig = cv2.imread(self._root_dir+im_file)

        img = self._image_process(img_orig, self._fixed_image_size)
        img = img.reshape((1,) + img.shape)
        phoc = self._phoc[int(label)]
        phoc = torch.from_numpy(phoc)
        length = self.length_embeddings[int(label)]
        length = torch.from_numpy(length)
        isquery = 1
        
        return img, phoc, length,int(label),isquery

    def __len__(self):
        return len(self._data_set)

    def embedding_size(self):
        return len(self._phoc[0])
    def get_label(self):
        return self._lex

    @staticmethod
    # pre processing and size normalization
    def _image_process(word_img, fixed_img_size, iscolor=False, mean_file=None):

        if not iscolor:
            word_img = np.around(np.dot(word_img[..., :3], [0.2989, 0.5870, 0.1140]))
            word_img = np.array(word_img, dtype=np.uint8)

        if mean_file is not None:
            word_img = word_img.astype(np.float32, copy=False)
            word_img = (word_img - np.mean(word_img)) / ((np.std(word_img) + 0.0001) / 128)
        else:
            word_img = word_img.astype(np.float32, copy=False)
            word_img = (word_img - np.mean(word_img)) / ((np.std(word_img) + 0.0001) / 128)

        if fixed_img_size is not None:
            if len(fixed_img_size) == 1:
                scale = float(fixed_img_size[0]) / float(word_img.shape[0])
                new_shape = (int(scale * word_img.shape[0]), int(scale * word_img.shape[1]))

            if len(fixed_img_size) == 2:
                new_shape = (fixed_img_size[0], fixed_img_size[1])

            word_img = cv2.resize(word_img, new_shape, interpolation=cv2.INTER_LINEAR).astype(np.float32)

        return word_img
