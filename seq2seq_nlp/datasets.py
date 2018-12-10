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

        #source, target = data['source'], data['target']
         # Same number of rows
        #append EOS token to both source and target
        assert len(data['source']) == len(data['target'])

        self.source, self.target = [], []
        for i in range(len(data['source'])):
            data['source'][i].append(3)
            data['target'][i].append(3)
            self.source.append(data['source'][i])
            self.target.append(data['target'][i])

        
        self.source_lens = [len(data['source'][i]) for i in range(len(data['source']))]
        self.target_lens = [len(data['target'][i]) for i in range(len(data['target']))]
        
        # self.source = self.source[:100]
        # self.source_lens = self.source_lens[:100]
        # self.target = self.target[:100]
        # self.target_lens = self.target_lens[:100]

        assert len(self.source) == len(self.target)
        #self.source, self.target = [], []
        #self.source_lens, self.target_lens = [], []

        # for i in range(len(source)):
        #     self.source.append(source[i][:max_len_source] + [PAD_IDX] * \
        #                        max(0, max_len_source - len(source[i])))

        #     self.target.append([SOS_IDX] + target[i][:max_len_target-2] + [EOS_IDX] + \
        #                             [PAD_IDX] * max(0, max_len_target - 2 - len(target[i])))

        #     self.source_lens.append(min(len(source[i]), max_len_source))

        #     true_len_target = len(target[i])
        #     if true_len_target > max_len_target-2:
        #         self.target_lens.append(max_len_target)
        #     else:
        #         self.target_lens.append(true_len_target+2)

        # self.source = torch.from_numpy(np.array(self.source))
        # self.target = torch.from_numpy(np.array(self.target))
        # self.source_lens = torch.from_numpy(np.array(self.source_lens))
        # self.target_lens = torch.from_numpy(np.array(self.target_lens))

    def __len__(self):
        return len(self.source)

    def __getitem__(self, ix):
        return self.source[ix], self.source_lens[ix], \
               self.target[ix], self.target_lens[ix]

def nmt_collate_fn_train(batch, max_len_source, max_len_target):
    #batch_size, x_L, y_L = len(batch), len(batch[0][0]), len(batch[0][2])
    #batch_sorted = sorted(batch, key=lambda x: x[1], reverse=True)

    # x = torch.LongTensor(batch_size, max_len_source)
    # x_lens = torch.LongTensor(batch_size, 1)
    # y = torch.LongTensor(batch_size, max_len_target)
    # y_lens = torch.LongTensor(batch_size, 1)
    padded_vec_source_batch = []
    padded_vec_target_batch = []
    source_len_batch = []
    target_len_batch = []
    for ix, (source, source_len, target, target_len) in enumerate(batch):
        source_len_batch.append(source_len)
        target_len_batch.append(target_len)

    max_len_source = max(source_len_batch) if max_len_source > max(source_len_batch) else max_len_source
    max_len_target = max(target_len_batch) if max_len_target > max(target_len_batch) else max_len_target 

    for ix, (source, source_len, target, target_len) in enumerate(batch):
        if source_len > max_len_source:
            padded_vec_source = np.array(source)[:max_len_source]
        else:
            padded_vec_source = np.pad(np.array(source),pad_width = ((0,max_len_source-source_len)),mode='constant', constant_values = 0)
        if target_len > max_len_target:
            padded_vec_target = np.array(target)[:max_len_target]
        else:
            padded_vec_target = np.pad(np.array(target),pad_width = ((0,max_len_target-target_len)),mode='constant', constant_values = 0)

        padded_vec_source_batch.append(padded_vec_source)
        padded_vec_target_batch.append(padded_vec_target)

    padded_vec_source_batch = np.array(padded_vec_source_batch)
    padded_vec_target_batch = np.array(padded_vec_target_batch)

    source_len_batch = np.array(source_len_batch)
    target_len_batch = np.array(target_len_batch)

    source_len_batch[source_len_batch > max_len_source] = max_len_source
    target_len_batch[target_len_batch > max_len_target] = max_len_target

    return [torch.from_numpy(padded_vec_source_batch).long(), torch.from_numpy(source_len_batch).unsqueeze(1).long(), 
            torch.from_numpy(padded_vec_target_batch).long(), torch.from_numpy(target_len_batch).unsqueeze(1).long()]


def nmt_collate_fn_val(batch, max_len_source, max_len_target):
    #for this the batch size will be 1 so we don't need to worry about it
    return [torch.from_numpy(np.array(batch[0][0])).unsqueeze(0).long(), torch.from_numpy(np.array(batch[0][1])).unsqueeze(0).unsqueeze(1).long(),
            torch.from_numpy(np.array(batch[0][2])).unsqueeze(0).long(), torch.from_numpy(np.array(batch[0][3])).unsqueeze(0).unsqueeze(1).long()]






    # for ix, (source, source_len, target, target_len) in enumerate(batch):
    #     x[ix, :] = source
    #     x_lens[ix, :] = source_len
    #     y[ix, :] = target
    #     y_lens[ix, :] = target_len

    # return x, x_lens, y, y_lens
