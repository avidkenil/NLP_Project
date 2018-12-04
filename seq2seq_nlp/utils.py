import numpy as np
import pandas as pd
import os
import logging
import time
import sys
import pickle
from pprint import pformat

import torch
from torch import nn
from torch.nn import functional
from torch.autograd import Variable

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Custom loss function with masked outputs till the sequence length
# taken from https://github.com/spro/practical-pytorch/tree/master/seq2seq-translation
# Made required changes for PyTorch 0.4 and integrating with our code.
def sequence_mask(sequence_length,device,max_len=None,):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.range(0, max_len - 1).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len).to(device)
    # seq_range_expand = Variable(seq_range_expand)
    # if sequence_length.is_cuda:
    #     seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand


def masked_cross_entropy(logits, target, length, device):

    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.
    Returns:
        loss: An average loss value masked by the length.
    """

    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = functional.log_softmax(logits_flat)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1), device=device)
    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum()
    return loss

def train(encoder, decoder, dataloader, criterion, optimizer, epoch, max_len_target, clip_param, device):
    loss_hist = []
    for batch_idx, (source, source_lens, target, target_lens) in enumerate(dataloader):
        source, source_lens  = source.to(device), source_lens.to(device)
        target, target_lens = target.to(device), target_lens.to(device)
        encoder.train()
        decoder.train()

        # # init the decoder outputs with zeros and then fill them up with the values
        # decoder_outputs = torch.zeros(source.size(0), max_len_target, decoder.vocab_size).to(device)
        encoder_output, encoder_hidden = encoder(source,source_lens)
        # doing complete teacher forcing first and then will add the probability based teacher forcing
        if decoder.num_layers == 1:
            if encoder.num_directions == 1:
                decoder_hidden_step = encoder_hidden[-2:]
            else:
                decoder_hidden_step = encoder_hidden[-1].unsqueeze(0)
        else:
            decoder_hidden_step = encoder_hidden

        input_seq = target[:,0]

        loss = 0.

        max_batch_target_len = target_lens.data.max().item()
        for step in range(max_batch_target_len):
            decoder_output_step, decoder_hidden_step, attn_weights_step = \
                decoder(input_seq, decoder_hidden_step, source_lens,
                        encoder_output)
            # decoder_outputs[:,step,:] = decoder_output_step
            input_seq = target[:,step] # change this line to change what to give as the next input to the decoder
            loss += criterion(decoder_output_step, input_seq)
        loss /= target_lens.data.sum().item() # Take per-element average
        #decoder.decode(decoder, target, target_lens, decoder_hidden_step,
                       #criterion=None)

        # loss = masked_cross_entropy(decoder_outputs[:,:max_batch_target_len,:].contiguous(),target,target_lens,device)
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        nn.utils.clip_grad_norm_(encoder.parameters(), clip_param)
        nn.utils.clip_grad_norm_(decoder.parameters(), clip_param)

        optimizer.step()

        # Accurately compute loss, because of different batch size
        loss_train = loss.item() * len(source) / len(dataloader.dataset)
        loss_hist.append(loss_train)

        if (batch_idx+1) % (len(dataloader.dataset)//(50*source.shape[0])) == 0:
            logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx+1) * source.shape[0], len(dataloader.dataset),
                100. * (batch_idx+1) / len(dataloader), loss.item()))

    optimizer.zero_grad()
    return loss_hist

def test(encoder, decoder, dataloader, criterion, device):
    encoder.eval()
    decoder.eval()

    loss_test = 0.
    y_hist = []
    output_hist = []
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            encoder_output = encoder(x)
            decoder_output = decoder(encoder_output)
            loss = criterion(decoder_output, y)

            # Accurately compute loss, because of different batch size
            loss_test += loss.item() / len(dataloader.dataset)

            output_hist.append(decoder_output)
            y_hist.append(y)

    return loss_test, torch.cat(output_hist, dim=0), torch.cat(y_hist, dim=0)

def print_config(vars_dict):
    vars_dict = {key: value for key, value in vars_dict.items() if key == key.upper()}
    logging.info(pformat(vars_dict))

def save_plot(project_dir, plots_dir, fig, filename):
    fig.savefig(os.path.join(project_dir, plots_dir, filename))

def make_dirs(parent_dir, child_dirs=None):
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    if child_dirs:
        for directory in child_dirs:
            directory_path = os.path.join(parent_dir, directory)
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)

def setup_logging(project_dir, logging_dir):
    log_path = os.path.join(project_dir, logging_dir)
    filename = '{}.log'.format(time.strftime('%Y_%m_%d'))
    log_handlers = [logging.FileHandler(os.path.join(log_path, filename)), logging.StreamHandler()]
    logging.basicConfig(format='%(asctime)s: %(message)s', datefmt='%Y/%m/%d %I:%M:%S %p', \
                        handlers=log_handlers, level=logging.DEBUG)
    logging.info('\n\n\n')

def save_object(object, filepath):
    '''
    This is a defensive way to write pickle.write, allowing for very large files on all platforms
    '''
    max_bytes = 2**31 - 1
    bytes_out = pickle.dumps(object, protocol=4)
    n_bytes = sys.getsizeof(bytes_out)
    with open(filepath, 'wb') as f_out:
        for idx in range(0, n_bytes, max_bytes):
            f_out.write(bytes_out[idx:idx+max_bytes])

def load_object(filepath):
    '''
    This is a defensive way to write pickle.load, allowing for very large files on all platforms
    '''
    max_bytes = 2**31 - 1
    try:
        input_size = os.path.getsize(filepath)
        bytes_in = bytearray(0)
        with open(filepath, 'rb') as f:
            for _ in range(0, input_size, max_bytes):
                bytes_in += f.read(max_bytes)
        object = pickle.loads(bytes_in)
    except Exception as error:
        print(error)
        return None
    return object

def clean_paired_files(path1, path2, path_out1, path_out2):
    data1 = load_txt(path1)
    data2 = load_txt(path2)

    final_data1, final_data2 = [], []
    for row1, row2 in zip(data1, data2):
        if row1.strip() and row2.strip():
            final_data1.append(row1)
            final_data2.append(row2)

    dump_txt_data(final_data1, path_out1)
    dump_txt_data(final_data2, path_out2)

def load_txt(path, f=lambda x: x):
    '''
    1. loads data from text file <path> where each line is a sentence.
    2. splits each line into a list (by spaces) and applies a
       function <f> to each individual element
    '''
    with open(path, 'r') as fin:
        data = [f(line) for line in fin if line.strip()]
    return data

def load_raw_data(path):
    return load_txt(path, f=lambda x: x.strip().split())

def load_ind_data(path):
    return load_txt(path, f=lambda line: [int(x) for x in line.strip().split()])

def dump_txt_data(obj, path):
    with open(path, 'w') as fout:
        for line in obj:
            fout.write(line)

def dump_ind_data(obj, path):
    with open(path, 'w') as fout:
        for line in obj:
            s = ' '.join([str(x) for x in line]) + '\n'
            fout.write(s)

def save_checkpoint(encoder, decoder, optimizer, train_loss_history, val_loss_history, \
                    train_bleu_history, val_bleu_history, epoch, args, project_dir,
                    checkpoints_dir, is_parallel=False):

    state_dict = {
        'encoder_state_dict': encoder.module.state_dict() if is_parallel else encoder.state_dict(),
        'decoder_state_dict': decoder.module.state_dict() if is_parallel else decoder.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'train_loss_history': train_loss_history,
        'val_loss_history': val_loss_history,
        'train_bleu_history': train_bleu_history,
        'val_bleu_history': val_bleu_history
    }

    source_dataset = os.path.splitext(args.source_dataset)[0]
    target_dataset = os.path.splitext(args.target_dataset)[0]

    params = [source_dataset, target_dataset, args.source_vocab, args.target_vocab, \
              args.max_len_source, args.max_len_target, args.encoder, args.num_directions, \
              args.encoder_num_layers, args.decoder_num_layers, args.encoder_emb_size, \
              args.decoder_emb_size, args.encoder_hid_size, args.encoder_dropout, \
              args.decoder_dropout, args.decoder_hid_size]

    state_dict_name = 'state_dict' + '_{}'*len(params) + '_epoch{}.pkl'
    state_dict_name = state_dict_name.format(*params, epoch)
    state_dict_path = os.path.join(project_dir, checkpoints_dir, state_dict_name)
    logging.info('Saving checkpoint "{}"...'.format(state_dict_path))
    torch.save(state_dict, state_dict_path)
    logging.info('Done.')

def remove_checkpoint(args, project_dir, checkpoints_dir, epoch):
    source_dataset = os.path.splitext(source_dataset)[0]
    target_dataset = os.path.splitext(target_dataset)[0]

    params = [source_dataset, target_dataset, args.source_vocab, args.target_vocab, \
              args.max_len_source, args.max_len_target, args.encoder, args.num_directions, \
              args.encoder_num_layers, args.decoder_num_layers, args.encoder_emb_size, \
              args.decoder_emb_size, args.encoder_hid_size, args.encoder_dropout, \
              args.decoder_dropout, args.decoder_hid_size]


    state_dict_name = 'state_dict' + '_{}'*len(params) + '_epoch{}.pkl'
    state_dict_name = state_dict_name.format(*params, epoch)
    state_dict_path = os.path.join(project_dir, checkpoints_dir, state_dict_name)
    logging.info('Removing checkpoint "{}"...'.format(state_dict_path))
    if os.path.exists(state_dict_path):
        os.remove(state_dict_path)
    logging.info('Done.')

def load_checkpoint(encoder, decoder, optimizer, checkpoint_file, project_dir, checkpoints_dir, device):
    # Note: Input model & optimizer should be pre-defined. This routine only updates their states.

    train_loss_history, val_loss_history = [], []
    train_bleu_history, val_bleu_history = [], []
    epoch_trained = 0

    state_dict_path = os.path.join(project_dir, checkpoints_dir, checkpoint_file)

    if os.path.isfile(state_dict_path):
        logging.info('Loading checkpoint "{}"...'.format(state_dict_path))
        state_dict = torch.load(state_dict_path)

        # Extract last trained epoch from checkpoint file
        epoch_trained = int(os.path.splitext(checkpoint_file)[0].split('_epoch')[-1])
        assert epoch_trained == state_dict['epoch']

        encoder.load_state_dict(state_dict['encoder_state_dict'])
        decoder.load_state_dict(state_dict['decoder_state_dict'])
        optimizer.load_state_dict(state_dict['optimizer'])
        train_loss_history = state_dict['train_loss_history']
        val_loss_history = state_dict['val_loss_history']
        train_bleu_history = state_dict['train_bleu_history']
        val_bleu_history = state_dict['val_bleu_history']

        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        logging.info('Successfully loaded checkpoint "{}".'.format(state_dict_path))

    else:
        raise FileNotFoundError('No checkpoint found at "{}"!'.format(state_dict_path))

    return encoder, decoder, optimizer, train_loss_history, val_loss_history, \
            train_bleu_history, val_bleu_history, epoch_trained

def save_model(model, model_name, epoch, args, project_dir, checkpoints_dir):
    params = [source_dataset, target_dataset, args.source_vocab, args.target_vocab, \
              args.max_len_source, args.max_len_target, args.encoder, args.num_directions, \
              args.encoder_num_layers, args.decoder_num_layers, args.encoder_emb_size, \
              args.decoder_emb_size, args.encoder_hid_size, args.encoder_dropout, \
              args.decoder_dropout, args.decoder_hid_size]

    checkpoint_name = model_name + '_{}'*len(params) + '_epoch{}.pt'
    checkpoint_name = checkpoint_name.format(*params, epoch)
    checkpoint_path = os.path.join(project_dir, checkpoints_dir, checkpoint_name)
    logging.info('Saving checkpoint "{}"...'.format(checkpoint_path))
    torch.save(model.to('cpu'), checkpoint_path)
    logging.info('Done.')

def load_model(project_dir, checkpoints_dir, checkpoint_file):
    checkpoint_path = os.path.join(project_dir, checkpoints_dir, checkpoint_file)
    if os.path.exists(checkpoint_path):
        logging.info('Saving checkpoint "{}"...'.format(checkpoint_path))
        model = torch.load(checkpoint_path)
        logging.info('Done.')

        # Extract last trained epoch from checkpoint file
        epoch_trained = int(os.path.splitext(checkpoint_file)[0].split('_epoch')[-1])

    else:
        raise FileNotFoundError('No checkpoint found at "{}"!'.format(checkpoint_path))

    return model, epoch_trained


class EarlyStopping(object):
    '''
    Implements early stopping in PyTorch
    Reference: https://gist.github.com/stefanonardo/693d96ceb2f531fa05db530f3e21517d
    '''

    def __init__(self, mode='minimize', min_delta=0, patience=10):
        self.mode = mode
        self._check_mode()
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.min_delta = min_delta

        if patience == 0:
            self.is_better = lambda metric: True
            self.stop = lambda metric: False

    def _check_mode(self):
        if self.mode not in {'maximize', 'minimize'}:
            raise ValueError('mode "{}" is unknown!'.format(self.mode))

    def is_better(self, metric):
        if self.best is None:
            return True
        if self.mode == 'minimize':
            return metric < self.best - self.min_delta
        return metric > self.best + self.min_delta

    def stop(self, metric):
        if self.best is None:
            self.best = metric
            return False

        if np.isnan(metric):
            return True

        if self.is_better(metric):
            self.num_bad_epochs = 0
            self.best = metric
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True
        return False
