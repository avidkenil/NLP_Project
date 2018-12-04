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
                 num_layers, bidirectional, return_type='full_last_layer',
                 dropout=0.5, device='cpu'):
        '''
        Arg in:
            kind (str): one of: 'rnn', 'gru', 'lstm'
            bidirectional (bool)
            return_type (str):
                'last_time_step' or 'full_last_layer'.
                If 'last_time_step' returns (B, L * D, H) tensor that is the output of the last
                    recurrent function of each layer in both directions
                If 'full_last_layer' returns (B, T, D * H) tensor that is the hidden states
                    from all time-steps of the last layer
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
        self.return_type = return_type

        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.rnn = self.rnn(embed_size, hidden_size, num_layers,
                            bidirectional=bidirectional, batch_first=True,
                            dropout=dropout)
        self.device = device

    def forward(self, x, x_lens):
        batch_size = x.size(0)

        embed = self.embedding(x)
        embed = pack_padded_sequence(embed, x_lens.squeeze(dim=1).long(),
                                     batch_first=True)
        out, h_n = self.rnn(embed, self._init_state(batch_size))
        out, _ = pad_packed_sequence(out, batch_first=True,
                                     total_length=x.size(1))
        # if self.return_type == 'last_time_step':
        #     out = h_n
        # NOTE: h_n will always be of shape (num_layers*num_directions, batch, hidden_size)
        if self.num_directions == 2:
            h_n = self._cat_hidden(h_n)
        return out, h_n

    def _init_state(self, batch_size):
        return torch.randn(self.num_layers * self.num_directions, batch_size,
                           self.hidden_size).to(self.device)

    def _cat_hidden(self, h_n):
        # if bidirectional then have to reshape hidden from
        # (num_layers*num_directions, batch, hidden_size) -> (num_layers, batch, hidden_size*num_directions)
        # currently just works for gru, and we will assume that we will just test for gru
        h_n = torch.cat([h_n[0:h_n.size(0):2], h_n[1:h_n.size(0):2]], dim=2)
        return h_n



class CNNEncoder(nn.Module):

    def __init__(self, vocab_size, embed_size, hidden_size, kernel_sizes, num_layers, \
                 dropout=0.5, dropout_type='1d'):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.kernel_sizes = kernel_sizes

        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)

        dropout = nn.Dropout if dropout_type == '1d' else nn.Dropout2d
        # TODO: implement dropout2d
        dropout = nn.Dropout
        cnn_sequences = []
        for kernel_size in kernel_sizes:
            cnn_sequence = []
            for layer in range(num_layers):
                in_channels = embed_size if layer == 0 else hidden_size
                cnn_sequence.extend(
                    [nn.Conv1d(in_channels, hidden_size, kernel_size,
                               padding=kernel_size // 2),
                     nn.ReLU(),
                     dropout(dropout)])
            cnn_sequences.append(nn.Sequential(*cnn_sequence))

        self.cnn_sequences = nn.ModuleList(cnn_sequences)

    def forward(self, x, x_lens):
        embed = self.embedding(x).transpose(1, 2)  # B, H, T
        outputs = []
        for ix, cnn in enumerate(self.cnn_sequences):
            out = cnn(embed).transpose(1,2)  # (B, T, H)
            to_add = self.num_layers if self.kernel_sizes[ix] % 2 == 0 else 0
            assert(out.size() == (x.size(0), x.size(1) +
                                  to_add, self.hidden_size))
            outputs.append(out)

        if self.num_layers == 1:
            # max-pooling
            for ix, output in enumerate(outputs):
                outputs[ix] = output.max(dim=1)[0]
        else:
            # avg-pooling
            for ix, output in enumerate(outputs):
                outputs[ix] = output.sum(dim=1) / x_lens  # (B, H)

        out = torch.cat(outputs, dim=1)
        assert(out.size() == (x.size(0),
                              len(self.kernel_sizes) * self.hidden_size))
        return out
