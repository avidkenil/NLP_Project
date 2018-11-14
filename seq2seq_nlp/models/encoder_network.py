import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNNEncoder(nn.Module):

    def __init__(self, kind, vocab_size, embed_size, hidden_size, num_layers,
                 bidirectional, return_type='full_last_layer',
                 p_dropout_hidden=0):
        '''
        Notation:
            B: batch size
            T: length of sequence
            E: embedding size
            H: hidden size
            D: num_directions
            L: num_layers

        Arg in:
            kind (str): one of: 'rnn', 'gru', 'lstm'
            bidirectional (bool)
            return_type (str):
                'last_time_step' or 'full_last_layer'.
                If 'last_time_step' returns (B, L * D, H) tensor that is the output of the last recurrent function of each layer in both directions
                If 'full_last_layer' returns (B, T, D * H) tensor that is the hidden states from all time-steps of the last layer
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

        self.table_lookup = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.rnn = self.rnn(embed_size, hidden_size, num_layers,
                            bidirectional=bidirectional, batch_first=True,
                            dropout=p_dropout_hidden)

    def init_state(self, batch_size):
        return torch.randn(self.num_layers * self.num_directions, batch_size,
                           self.hidden_size)

    def forward(self, x, x_len):
        batch_size = x.size(0)

        embed = self.table_lookup(x)
        embed = pack_padded_sequence(embed, x_lens.squeeze(dim=1).long(),
                                     batch_first=True)
        out, h_n = self.rnn(embed, self.init_state(batch_size))
        out, _ = pad_packed_sequence(out, batch_first=True,
                                     total_length=x.size(1))
        if self.return_type == 'last_time_step':
            out = h_n
        return out


if __name__ == '__main__':

    print('tests')
    x_lens = torch.FloatTensor([[5],[4],[4],[2]])
    V, B, T, H = 5000, 4, 5, 500

    x = torch.randint(0, V, size=(B, T)).long()

    gru = RNNEncoder('gru', V, 500, H, 3, True, p_dropout_hidden=.5)
    out = gru(x, x_lens)
    assert(out.size() == (B, T, 2 * H))
