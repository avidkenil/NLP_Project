import torch
import torch.nn as nn


class RNNDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, encoder_directions=1, encoder_hidden_size=256, \
                 num_layers=1, fc_hidden_size=512, attn=None, dropout=0.5):
        super(RNNDecoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.vocab_size = vocab_size
        self.attn = attn
        self.num_layers = num_layers
        self.rnn = nn.GRU(embed_size, hidden_size=encoder_directions*encoder_hidden_size, \
                          batch_first=True, num_layers=self.num_layers)
        self.fc1 = nn.Linear(encoder_directions*encoder_hidden_size, vocab_size)
        # self.fc2 = nn.Linear(fc_hidden_size, vocab_size)

    def forward(self, x, decoder_hidden, source_lens=None, encoder_outputs=None):
        # x: Bx1, decoder_hidden -> num_layers, B, encoder_hidden_size*num_directions
        x = x.unsqueeze(1)
        emb = self.embedding(x)

        output, hidden = self.rnn(emb, decoder_hidden)

        output = output.squeeze(1) # output: B x 1 x H -> B x H
        output = self.fc1(output)
        attn_weights = None
        return output, hidden, attn_weights
