import cv2
import numpy as np
# from phoc import build_phoc_descriptor, get_most_common_n_grams
import torch
from torch.utils.data import Dataset


class SynthDataSet(Dataset):
    def __init__(self, root_dir, dset='train', fixed_image_size=(100,32), randomize=True):
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
            self._data_set = [self._data_set[i] for i in random_index[:16000]]

        self._len = [len(w) for w in self._lex]

        unigrams = [chr(i) for i in list(range(ord('a'), ord('z') + 1)) + list(range(ord('0'), ord('9') + 1))]
        indices = range(37)
        unigram_dict = dict(zip(unigrams, indices))
        unigram_dict['-'] = 37
        unigram_dict['*'] = 38
        embeddings = []
        for elem in self._lex:
            y=[]
            for ch in elem+'*':
                try:
                    x = unigram_dict[ch]
                except:
                    x = 37
                y.append(x)
            embeddings.append(y)

        self.word_embeddings = embeddings
    def embedding_size(self):
        return len(self.word_embeddings[0])

    def __getitem__(self, index):

        im_file, label = self._data_set[index].split()
        # read image
        # print im_file
        # print label
        img = self._image_process(cv2.imread(self._root_dir + im_file), self._fixed_image_size)
        img = img.reshape((1,) + img.shape)
        img = torch.from_numpy(img)
        label = self.word_embeddings[int(label)]
        # print(label)
        # print self._lex[int(label)]
        # phoc = self._phoc[int(label)]
        # length = self._len[int(label)]
        return img, label
            # , phoc, length

    def __len__(self):
        return len(self._data_set)

    @staticmethod
    # pre processing and size normalization
    def _image_process(word_img, fixed_img_size):

        word_img =np.around(np.dot(word_img[...,:3], [0.2989, 0.5870, 0.1140]))
        word_img = np.array(word_img, dtype=np.uint8)

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
