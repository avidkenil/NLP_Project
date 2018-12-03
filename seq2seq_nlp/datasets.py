import numpy as np
import torch
import logging
from torch.utils.data import Dataset


class NMTDataset(Dataset):
    def __init__(self, data, max_len_source=300, max_len_target = 300, pad_idx=0):
        logging.info('Truncating to maximum lengths and '\
                      'padding all sequences with required zeros.')

        self.max_len_source = max_len_source
        self.max_len_target = max_len_target

        source, target = data['source'], data['target']
        self.source_lens = np.array([len(sent) if len(sent) < max_len_source \
                                     else max_len_source for sent in source], \
                                     dtype=np.int64)
        self.target_lens = np.array([len(sent) if len(sent) < max_len_target \
                                     else max_len_target for sent in target], \
                                     dtype=np.int64)

        self.source = np.array([
            sent[:self.max_len_source] + [pad_idx] * max(0, self.max_len_source - len(sent)) \
            for sent in source], dtype=np.int64)
        self.target = np.array([
            sent[:self.max_len_target] + [pad_idx] * max(0, self.max_len_target - len(sent)) \
            for sent in target], dtype=np.int64)
        assert len(self.source) == len(self.target) # Same number of rows

        self.source = torch.from_numpy(self.source)
        self.target = torch.from_numpy(self.target)
        self.source_lens = torch.from_numpy(self.source_lens)
        self.target_lens = torch.from_numpy(self.target_lens)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, ix):
        return self.source[ix], self.source_lens[ix], \
               self.target[ix], self.target_lens[ix]

def nmt_collate_fn(batch, max_len_source, max_len_target):
    batch_size, x_L, y_L = len(batch), len(batch[0][0]), len(batch[0][2])

    batch_sorted = sorted(batch, key=lambda x: x[1], reverse=True)
    source_lens = np.array([batch[i][1] for i in range(len(batch))])
    target_lens = np.array([batch[i][3] for i in range(len(batch))])

    # Get the max lengths of the source and the target
    max_batch_len_source = min(max_len_source,max(source_lens).item())
    max_batch_len_target = min(max_len_target,max(target_lens).item())

    x = torch.LongTensor(batch_size, max_batch_len_source)
    x_lens = torch.FloatTensor(batch_size, 1)
    y = torch.LongTensor(batch_size, max_batch_len_target)
    y_lens = torch.FloatTensor(batch_size, 1)

    for ix, (source, source_len, target, target_len) in enumerate(batch_sorted):
        x[ix, :] = source[:max_batch_len_source]
        x_lens[ix, :] = source_len
        y[ix, :] = target[:max_batch_len_target]
        y_lens[ix, :] = target_len

    return x.long(), x_lens.long(), y.long(), y_lens.long()
