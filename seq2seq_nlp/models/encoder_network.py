import torch
import torch.nn as nn

class RNNEncoder(nn.Module):
    def __init__(self, kind, hidden_size, num_layers, bidirectional):
        '''
        kind (str): one of: 'rnn', 'gru', 'lstm'
        '''
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1

    def init_state(self, batch_size):
        return torch.rand(batch_size, self.hidden_size)

    def forward(self, x, x_len):
        B, L = x.size()
        return torch.rand(B, L, self.num_directions * self.hidden_size)

if __name__ == '__main__':
    B, L, H = 32, 300, 500
    x = torch.rand(B, L)
    x_lens = torch.randint(1, L+1, (B,1))
    gru = RNN('gru', H, 3, True)
    out = gru(x, x_lens)
    assert(out.size() == (B, L, 2 * H))

