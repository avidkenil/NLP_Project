import numpy as np
import pandas as pd
import os
import logging
import time
import sys
import pickle
from pprint import pformat

import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def train(encoder, decoder, dataloader, criterion, optimizer, device, epoch):
    loss_hist = []
    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.to(device).float(), y.to(device).long()
        encoder.train()
        decoder.train()

        optimizer.zero_grad()

        encoder_output = encoder(x)
        decoder_output = decoder(encoder_output)

        loss = criterion(decoder_output, y)
        loss.backward()
        optimizer.step()

        # Accurately compute loss, because of different batch size
        loss_train = loss.item() * len(x) / len(dataloader.dataset)
        loss_hist.append(loss_train)

        if (batch_idx+1) % (len(dataloader.dataset)//(5*y.shape[0])) == 0:
            logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx+1) * y.shape[0], len(dataloader.dataset),
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
            x, y = x.to(device).float(), y.to(device).long()
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

def make_dirs(parent_dir, directories_to_create):
    for directory in directories_to_create:
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

def load_txt(path, f=lambda x: x):
    '''
    1. loads data from text file <path> where each line is a sentence.
    2. splits each line into a list (by spaces) and applies a
       function <f> to each individual element
    '''
    with open(path, 'r') as fin:
        data = [f(line) for line in fin if line]
    return data


def load_raw_data(path):
    return load_txt(path, f=lambda x: x.strip().split())


def load_ind_data(path):
    return load_txt(path, f=lambda line: [int(x) for x in line.strip().split()])


def dump_ind_data(obj, path):
    with open(path, 'w') as fout:
        for line in obj:
            s = ' '.join([str(x) for x in line]) + '\n'
            fout.write(s)

def save_checkpoint(encoder, decoder, optimizer, train_loss_history, val_loss_history, \
                    train_accuracy_history, val_accuracy_history, epoch, source_dataset, \
                    target_dataset, project_dir, checkpoints_dir, is_parallel=False):

    state_dict = {
        'encoder_state_dict': encoder.module.state_dict() if is_parallel else encoder.state_dict(),
        'decoder_state_dict': decoder.module.state_dict() if is_parallel else decoder.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'train_loss_history': train_loss_history,
        'val_loss_history': val_loss_history,
        'train_accuracy_history': train_accuracy_history,
        'val_accuracy_history': val_accuracy_history
    }

    source_dataset = os.path.splitext(source_dataset)[0]
    target_dataset = os.path.splitext(target_dataset)[0]

    state_dict_name = 'state_dict_{}_{}_epoch{}.pkl'.format(source_dataset, target_dataset, epoch)
    state_dict_path = os.path.join(project_dir, checkpoints_dir, state_dict_name)
    logging.info('Saving checkpoint "{}"...'.format(state_dict_path))
    torch.save(state_dict, state_dict_path)
    logging.info('Done.')

def remove_checkpoint(dataset, project_dir, checkpoints_dir, epoch):
    source_dataset = os.path.splitext(source_dataset)[0]
    target_dataset = os.path.splitext(target_dataset)[0]
    state_dict_name = 'state_dict_{}_{}_epoch{}.pkl'.format(source_dataset, target_dataset, epoch)
    state_dict_path = os.path.join(project_dir, checkpoints_dir, state_dict_name)
    logging.info('Removing checkpoint "{}"...'.format(state_dict_path))
    if os.path.exists(state_dict_path):
        os.remove(state_dict_path)
    logging.info('Done.')

def load_checkpoint(encoder, decoder, optimizer, checkpoint_file, project_dir, checkpoints_dir, device):
    # Note: Input model & optimizer should be pre-defined. This routine only updates their states.

    train_loss_history, val_loss_history = [], []
    train_accuracy_history, val_accuracy_history = [], []
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
        train_accuracy_history = state_dict['train_accuracy_history']
        val_accuracy_history = state_dict['val_accuracy_history']

        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        logging.info('Successfully loaded checkpoint "{}".'.format(state_dict_path))

    else:
        raise FileNotFoundError('No checkpoint found at "{}"!'.format(state_dict_path))

    return encoder, decoder, optimizer, train_loss_history, val_loss_history, \
            train_accuracy_history, val_accuracy_history, epoch_trained


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
