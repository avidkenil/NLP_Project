import numpy as np
import torch
import logging
from torch.utils.data import Dataset


PAD, PAD_IDX = '<pad>', 0
UNK, UNK_IDX = '<unk>', 1
SOS, SOS_IDX = '<sos>', 2
EOS, EOS_IDX = '<eos>', 3


class NMTDataset(Dataset):
    def __init__(self, data, max_len_source=100, max_len_target=100):
        logging.info('Truncating to maximum lengths and '\
                     'padding all sequences with required zeros')

        source, target = data['source'], data['target']
        assert len(source) == len(target) # Same number of rows

        self.source, self.target = [], []
        self.source_lens, self.target_lens = [], []

        for i in range(len(source)):
            self.source.append(source[i][:max_len_source] + [PAD_IDX] * \
                               max(0, max_len_source - len(source[i])))

            self.target.append([SOS_IDX] + target[i][:max_len_target-2] + [EOS_IDX] + \
                                    [PAD_IDX] * max(0, max_len_target - 2 - len(target[i])))

            self.source_lens.append(min(len(source[i]), max_len_source))

            true_len_target = len(target[i])
            if true_len_target > max_len_target-2:
                self.target_lens.append(max_len_target)
            else:
                self.target_lens.append(true_len_target+2)

        self.source = torch.from_numpy(np.array(self.source))
        self.target = torch.from_numpy(np.array(self.target))
        self.source_lens = torch.from_numpy(np.array(self.source_lens))
        self.target_lens = torch.from_numpy(np.array(self.target_lens))

    def __len__(self):
        return len(self.source)

    def __getitem__(self, ix):
        return self.source[ix], self.source_lens[ix], \
               self.target[ix], self.target_lens[ix]

def nmt_collate_fn(batch, max_len_source, max_len_target):
    batch_size, x_L, y_L = len(batch), len(batch[0][0]), len(batch[0][2])
    batch_sorted = sorted(batch, key=lambda x: x[1], reverse=True)

    x = torch.LongTensor(batch_size, max_len_source)
    x_lens = torch.LongTensor(batch_size, 1)
    y = torch.LongTensor(batch_size, max_len_target)
    y_lens = torch.LongTensor(batch_size, 1)

    for ix, (source, source_len, target, target_len) in enumerate(batch_sorted):
        x[ix, :] = source
        x_lens[ix, :] = source_len
        y[ix, :] = target
        y_lens[ix, :] = target_len

    return x, x_lens, y, y_lens
