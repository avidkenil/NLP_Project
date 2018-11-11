import pickle as pkl


def load_pickle(path):
    with open(path, 'rb') as fin:
        obj = pkl.load(fin)
    return obj


def dump_pickle(obj, path):
    with open(path, 'wb') as fout:
        obj = pkl.dump(obj, fout)


def load_txt(path, f=lambda x: x):
    '''
    1. loads data from text file <path> where each line is a sentence.
    2. splits each line into a list (by spaces) and applies a
       function <f> to each individual element
    '''
    with open(path, 'r') as fin:
        data = [f(line) for line in fin if line]
    return data


def load_raw_data(path):
    return load_txt(path, f=lambda x: x.strip().split())


def load_ind_data(path):
    return load_txt(path, f=lambda line: [int(x) for x in line.strip().split()])


def dump_ind_data(obj, path):
    with open(path, 'w') as fout:
        for line in obj:
            s = ' '.join([str(x) for x in line]) + '\n'
            fout.write(s)
