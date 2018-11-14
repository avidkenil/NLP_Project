import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import utils
from preprocessing import PAD

def get_loader(source_path, target_path, batch_size=64, shuffle=True):
    '''
    data_path (str): path of pickled indexed data
    '''
    data = MT_Dataset(source_path, target_path)
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle,
                      num_workers=1, collate_fn=collate_fn)

def collate_fn(datum):
    batch_size, x_L, y_L = len(datum), len(datum[0][0]), len(datum[0][2])

    datum_sorted = sorted(datum, key=lambda x: x[1], reverse=True)

    x = torch.LongTensor(batch_size, x_L)
    x_lens = torch.FloatTensor(batch_size, 1)
    y = torch.LongTensor(batch_size, y_L)
    y_lens = torch.FloatTensor(batch_size, 1)

    for ix, (source, source_len, target, target_len) in enumerate(datum_sorted):
        x[ix, :] = source
        x_lens[ix, :] = source_len
        y[ix, :] = target
        y_lens[ix, :] = target_len

    return x, x_lens, y, y_lens

class MT_Dataset(Dataset):

    max_len = 300

    def __init__(self, source_path, target_path, tok2ind=None):
        '''
        path (str): path of data of indices
        '''
        pad_ind = tok2ind[PAD] if tok2ind else 0
        source = utils.load_ind_data(source_path)
        target = utils.load_ind_data(target_path)

        self.source_lens = np.array([len(sent) for sent in source],
                                    dtype=np.float)
        self.target_lens = np.array([len(sent) for sent in target],
                                    dtype=np.float)
        self.source = np.array([
            sent[:self.max_len] + [pad_ind] * max(0, self.max_len - len(sent))
            for sent in source], dtype=np.int64)
        self.target = np.array([
            sent[:self.max_len] + [pad_ind] * max(0, self.max_len - len(sent))
            for sent in target], dtype=np.int64)
        assert(len(self.source) == len(self.target))

        self.source = torch.from_numpy(self.source)
        self.target = torch.from_numpy(self.target)
        self.source_lens = torch.from_numpy(self.source_lens)
        self.target_lens = torch.from_numpy(self.target_lens)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, ix):
        return self.source[ix], self.source_lens[ix], self.target[ix],\
            self.target_lens[ix]


if __name__ == '__main__':
    val_loader = get_loader('../data/vi-en/dev.tok.ind.50000.vi', '../data/vi-en/dev.tok.ind.50000.en', shuffle=False)

    for s, sl, t, tl in val_loader:
        print(f"s: {s.size()}; {s.type()}")
        print(f"sl: {sl.size()}; {sl.type()}")
        print(f"t: {t.size()}; {t.type()}")
        print(f"tl: {tl.size()}; {tl.type()}")
        break
