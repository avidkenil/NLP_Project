import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionModule(nn.Module):
    def __init__(self, input_dim, output_dim, device):
        super(AttentionModule,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.device = device
        self.Wa = nn.Linear(self.input_dim,self.output_dim, bias = False)
        self.Wc = nn.Linear(self.input_dim+self.output_dim,self.output_dim, bias = False)


    def forward(self,decoder_hidden,encoder_outputs,source_lens):
        '''
        decoder_hidden = bz x input_dim(decoder hidden dim)
        encoder_outputs = bz x src_len x output_dim(encoder output dim which is same as input dim)
        '''
        # B x out_dim
        x = self.Wa(decoder_hidden)
        # B x src_len
        attn_scores = (encoder_outputs.transpose(0,1)*x.unsqueeze(0)).sum(dim = 2)
        # masking the scores for the padded tokens
        enc_masks = get_mask(source_lens,self.device,source_lens.max().item()).transpose(0,1)
        masked_attn = enc_masks.float()*attn_scores
        masked_attn[masked_attn == 0] = -1e10
        attn_scores = F.softmax(masked_attn,dim=0)

        x = (attn_scores.unsqueeze(2)*encoder_outputs.transpose(0,1)).sum(dim=0)
        x = F.tanh(self.Wc(torch.cat((x,decoder_hidden),dim=1)))

        return x, attn_scores
