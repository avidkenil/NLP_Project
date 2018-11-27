import torch
import torch.nn as nn




class RNNDecoder(nn.Module):-
    def __init__(self, vocab_size, embed_size, p_dropoutfc_hidden_size = 512, attn=None,encoder_directions = 1,encoder_hidden_size=256):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size,embed_size,padding_idx=0)
        self.attn = attn
        self.rnn = nn.GRU(embed_size,hidden_size=encoder_directions*encoder_hidden_size,batch_first=True)
        self.fc1 = nn.Linear(hidden_size,vocab_size)
        #self.fc2 = nn.Linear(fc_hidden_size, vocab_size)

    def forward(input, decoder_hidden, source_lens=None, encoder_outputs=None):
        #input-> Bx1,decoder_hidden -> Bxenc_hidden_size*num_directions
        emb = self.embedding(input)
        output,hidden = self.rnn(emb,decoder_hidden)
        #output B x 1 x H -> B x H
        output = output.squeeze(1)
        output = self.fc1(output)
        attn_weights = None
        return output,hidden,attn_weights
