import xmltodict
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def extract_data(path =None, lex =None):

    if path is not None:
        test_file = path+'test.xml'
        with open(test_file) as fd:
            doc = xmltodict.parse(fd.read())
        # print doc
    if lex is None:
        full_lex = []
        for images in  doc['tagset']['image']:
            full_lex.extend(images['lex'].split(','))
        full_lex = np.unique(full_lex)
    else:
        full_lex = lex
    lens = [len(x) for x in full_lex]
    maxlen = max(lens)
    x_i=[]
    for i in range(1,maxlen):
        x_i.append(len([x for x in lens if x >= i]))

    x_i = [x/np.float(len(full_lex)) for x in x_i]
    print x_i
    return x_i, maxlen






if __name__ == '__main__':

    x_i_svt,maxlen_svt = extract_data(path='/home/suman/dataset/svt/')
    f=open('lex_iam.txt', 'r')
    iam_lex = f.readlines()
    x_i_iam, maxlen_iam = extract_data(lex=iam_lex)
    maxlen = max(maxlen_iam,maxlen_svt)
    for i in range(maxlen_svt,maxlen_iam):
        x_i_svt.append(0.0)


    plt.xticks(range(1, maxlen))
    plt.plot(range(1, maxlen), x_i_svt)
    plt.plot(range(1, maxlen), x_i_iam)
    plt.savefig('len_stat.pdf')
    plt.show()