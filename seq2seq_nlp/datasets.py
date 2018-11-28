import numpy as np
import torch
import logging
from torch.utils.data import Dataset


class NMTDataset(Dataset):
    def __init__(self, data, max_len=300, pad_idx=0):
        '''
        path (str): path of data of indices
        '''
        self.max_len = max_len

        source, target = data['source'], data['target']
        self.source_lens = np.array([len(sent) for sent in source], dtype=np.float)
        self.target_lens = np.array([len(sent) for sent in target], dtype=np.float)

        self.source = np.array([
            sent[:self.max_len] + [pad_idx] * max(0, self.max_len - len(sent)) \
            for sent in source], dtype=np.int64)
        self.target = np.array([
            sent[:self.max_len] + [pad_idx] * max(0, self.max_len - len(sent)) \
            for sent in target], dtype=np.int64)
        assert(len(self.source) == len(self.target))

        self.source = torch.from_numpy(self.source)
        self.target = torch.from_numpy(self.target)
        self.source_lens = torch.from_numpy(self.source_lens)
        self.target_lens = torch.from_numpy(self.target_lens)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, ix):
        return self.source[ix], self.source_lens[ix], \
               self.target[ix], self.target_lens[ix]

def nmt_collate_fn(batch):
    batch_size, x_L, y_L = len(batch), len(batch[0][0]), len(batch[0][2])

    batch_sorted = sorted(batch, key=lambda x: x[1], reverse=True)

    x = torch.LongTensor(batch_size, x_L)
    x_lens = torch.FloatTensor(batch_size, 1)
    y = torch.LongTensor(batch_size, y_L)
    y_lens = torch.FloatTensor(batch_size, 1)

    for ix, (source, source_len, target, target_len) in enumerate(batch_sorted):
        x[ix, :] = source
        x_lens[ix, :] = source_len
        y[ix, :] = target
        y_lens[ix, :] = target_len

    return x.long(), x_lens.long(), y.long(), y_lens.long()
