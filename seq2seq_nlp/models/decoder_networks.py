import torch
import torch.nn as nn
import torch.nn.functional as F

from seq2seq_nlp.attention import *


def get_mask(source_lens, device, max_len=None):
    if max_len is None:
        max_len = source_lens.max().item()
    batch_size = source_lens.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).repeat([batch_size,1]).to(device)
    seq_length_expand = (source_lens.expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand


class RNNDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, kind='gru',encoder_directions=1, encoder_hidden_size=256, encoder_type='cnn',\
                 num_layers=1, fc_hidden_size=512, attn=False, dropout=0.1, joint_hidden_ec=False,device='cpu'):
        super(RNNDecoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.joint_hidden_ec = joint_hidden_ec
        self.hidden_size = encoder_hidden_size
        if encoder_type != 'cnn':
            self.hidden_size = encoder_directions*encoder_hidden_size*2 if joint_hidden_ec else encoder_directions*encoder_hidden_size
        if kind == 'gru':
            self.rnn = nn.GRU
        elif kind == 'lstm':
            self.rnn = nn.LSTM
        elif kind == 'rnn':
            self.rnn = nn.RNN
        else:
            raise NotImplementedError

        self.fc1 = nn.Linear(encoder_directions*encoder_hidden_size, vocab_size)
        self.device = device
        self.dropout = nn.Dropout(p=0.1)
        self.attn = AttentionModule(self.hidden_size, self.hidden_size, device=device) if attn else None
        if self.attn:
            self.rnn = self.rnn(self.hidden_size + embed_size, self.hidden_size, \
                    batch_first=True, num_layers=self.num_layers,dropout=0.1)

        else:
            self.rnn = self.rnn(embed_size, hidden_size=self.hidden_size, \
                          batch_first=True, num_layers=self.num_layers,dropout=0.1)

    def forward(self, x, decoder_hidden, source_lens=None, encoder_outputs=None,encoder_hidden = None,context_vec = None):
        # x: B, decoder_hidden -> num_layers, B, encoder_hidden_size*num_directions

        x = x.unsqueeze(1)
        emb = self.embedding(x)
        emb = self.dropout(emb)
        if self.joint_hidden_ec:
            decoder_hidden = torch.cat([encoder_hidden,decoder_hidden],dim=2)
        if self.attn:
            emb = torch.cat((emb,context_vec.unsqueeze(1)),dim=2)

        output, hidden = self.rnn(emb, decoder_hidden)

        output = output.squeeze(1) # output: B x 1 x H -> B x H
        output = self.dropout(output)
        if self.attn:
            context_vec, attn_scores = self.attn(output,encoder_outputs,source_lens)
            output = self.fc1(context_vec)
        else:
            output = self.fc1(output)
            context_vec = None
            attn_scores = None

        return F.log_softmax(output,dim=1), hidden, attn_scores, context_vec


    def _init_state(self, batch_size):
        if isinstance(self.rnn,nn.LSTM):
            return(torch.zeros(self.num_layers * self.num_directions, batch_size,
                           self.hidden_size).to(self.device),
                   torch.zeros(self.num_layers * self.num_directions, batch_size,
                           self.hidden_size).to(self.device))
        else:
            return torch.zeros(self.num_layers * self.num_directions, batch_size,
                           self.hidden_size).to(self.device)

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
