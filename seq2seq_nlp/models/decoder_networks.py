import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, kind='gru',encoder_directions=1, encoder_hidden_size=256, \
                 num_layers=1, fc_hidden_size=512, attn=None, dropout=0.1,joint_hidden_ec = False,device='cpu'):
        super(RNNDecoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.vocab_size = vocab_size
        self.attn = attn
        self.num_layers = num_layers
        self.joint_hidden_ec = joint_hidden_ec
        self.hidden_size = encoder_directions*encoder_hidden_size*2 if joint_hidden_ec else encoder_directions*encoder_hidden_size
        if kind == 'gru':
            self.rnn = nn.GRU
        elif kind == 'lstm':
            self.rnn = nn.LSTM
        elif kind == 'rnn':
            self.rnn = nn.RNN
        else:
            raise NotImplementedError

        self.rnn = self.rnn(embed_size, hidden_size=self.hidden_size, \
                          batch_first=True, num_layers=self.num_layers,dropout=0.1)
        self.fc1 = nn.Linear(encoder_directions*encoder_hidden_size, vocab_size)
        self.device = device
        self.dropout = nn.Dropout(p=0.1)
        # self.fc2 = nn.Linear(fc_hidden_size, vocab_size)

    def forward(self, x, decoder_hidden, source_lens=None, encoder_outputs=None,encoder_hidden = None):
        # x: B, decoder_hidden -> num_layers, B, encoder_hidden_size*num_directions

        x = x.unsqueeze(1)
        emb = self.embedding(x)
        emb = self.dropout(emb)
        if self.joint_hidden_ec:
            decoder_hidden = torch.cat([encoder_hidden,decoder_hidden],dim=2)
        output, hidden = self.rnn(emb, decoder_hidden)

        output = output.squeeze(1) # output: B x 1 x H -> B x H
        output = self.dropout(output)
        output = self.fc1(output)
        attn_weights = None
        return F.log_softmax(output,dim=1), hidden, attn_weights


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


