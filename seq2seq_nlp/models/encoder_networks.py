# Notation in the comments throughout:
# B: batch size
# T: length of sequence
# E: embedding size
# H: hidden size
# D: num_directions
# L: num_layers

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class RNNEncoder(nn.Module):

    def __init__(self, kind, vocab_size, embed_size, hidden_size,
                 num_layers, bidirectional,
                 dropout=0.2, device='cpu'):
        '''
        kind (str): one of: 'rnn', 'gru', 'lstm'
        '''

        super().__init__()
        if kind == 'rnn':
            self.rnn = nn.RNN
        elif kind == 'gru':
            self.rnn = nn.GRU
        elif kind == 'lstm':
            self.rnn = nn.LSTM

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1

        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.rnn = self.rnn(embed_size, hidden_size, num_layers,
                            bidirectional=bidirectional, batch_first=True,
                            dropout=dropout)
        self.device = device
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x, x_lens):
        batch_size = x.size(0)
        sorted_idx = sorted(range(len(x_lens)), key=lambda x: -x_lens[x])
        orig_idx = sorted(range(len(x_lens)), key=lambda x: sorted_idx[x])
        embed = self.embedding(x)
        embed = self.dropout(embed)
        embed = embed[sorted_idx]
        sorted_lens = x_lens[sorted_idx]
        embed = pack_padded_sequence(embed, sorted_lens.squeeze(dim=1).long(),
                                     batch_first=True)
        out, h_n = self.rnn(embed, self._init_state(batch_size))
        out, _ = pad_packed_sequence(out, batch_first=True,padding_value=0)
        # NOTE: h_n will always be of shape (num_layers*num_directions, batch, hidden_size)
        if self.num_directions == 2:
            h_n = self._cat_hidden(h_n,batch_size,orig_idx)
        return out[orig_idx], h_n

    def _init_state(self, batch_size):
        if isinstance(self.rnn,nn.LSTM):
            return(torch.zeros(self.num_layers * self.num_directions, batch_size,
                           self.hidden_size).to(self.device),
                   torch.zeros(self.num_layers * self.num_directions, batch_size,
                           self.hidden_size).to(self.device))
        else:
            return torch.zeros(self.num_layers * self.num_directions, batch_size,
                           self.hidden_size).to(self.device)

    def _cat_hidden(self, h_n,batch_size,orig_idx):
        # If bidirectional then have to reshape hidden from
        # (num_layers*num_directions, batch, hidden_size) -> (num_layers, batch, hidden_size*num_directions)
        # currently only works for gru, and we will assume that we will just test for gru
        if isinstance(self.rnn,nn.LSTM):
            return (h_n[0].view(self.num_layers, 2, batch_size, -1).transpose(1, 2).contiguous()\
                    .view(self.num_layers, batch_size, -1)[:,orig_idx,:],
                    h_n[1].view(self.num_layers, 2, batch_size, -1).transpose(1, 2).contiguous().\
                    view(self.num_layers, batch_size, -1)[:,orig_idx,:])
        else:
            return h_n.view(self.num_layers, 2, batch_size, -1).transpose(1, 2).contiguous().view(self.num_layers, batch_size, -1)[:,orig_idx,:]

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.LSTM) or isinstance(m, nn.GRU) or isinstance(m, nn.RNN):
                nn.init.xavier_normal_(m.weight)
                nn.init.uniform_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.uniform_(m.bias)



class CNNEncoder(nn.Module):

    def __init__(self, vocab_size, embed_size, hidden_size, kernel_sizes, num_layers, \
                 dropout=0.5, dropout_type='1d'):
        super().__init__()

        self.hidden_size = hidden_size
        self.kernel_sizes = kernel_sizes

        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)

        self.dropout = nn.Dropout(p=dropout) if dropout_type == '1d' else nn.Dropout2d
        cnn_sequences = []
        for kernel_size in kernel_sizes:
            cnn_sequence = []
            for layer in range(num_layers):
                in_channels = embed_size if layer == 0 else hidden_size
                cnn_sequence.extend(
                    [nn.Conv1d(in_channels, hidden_size, kernel_size,
                               padding=kernel_size // 2),
                     nn.ReLU(),
                     self.dropout(dropout)])
            cnn_sequences.append(nn.Sequential(*cnn_sequence))

        self.cnn_sequences = nn.ModuleList(cnn_sequences)

    def forward(self, x, x_lens):
        embed = self.embedding(x).transpose(1, 2)  # B, H, T
        outputs = []
        for ix, cnn in enumerate(self.cnn_sequences):
            out = cnn(embed).transpose(1,2)  # (B, T, H)
            outputs.append(out)

        # Max-pooling
        for ix, output in enumerate(outputs):
            outputs[ix] = output.max(dim=1)[0]

        out = torch.cat(outputs, dim=1)
        return None, out
