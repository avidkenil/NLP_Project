import numpy as np
import pandas as pd
import os
import argparse
import logging

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from seq2seq_nlp.arguments import get_args
from seq2seq_nlp.preprocessing import generate_dataloader
from seq2seq_nlp.models.encoder_networks import RNNEncoder
from seq2seq_nlp.models.decoder_networks import RNNDecoder
# from seq2seq_nlp.models.attention import VanillaAttention
from seq2seq_nlp.utils import *


args = get_args()


# Globals
PROJECT_DIR = args.project_dir
DATA_DIR,  PLOTS_DIR, LOGGING_DIR = args.data_dir, 'plots', 'logs'
CHECKPOINTS_DIR, CHECKPOINT_FILE = args.checkpoints_dir, args.load_ckpt
SOURCE_DATASET, TARGET_DATASET = args.source_dataset, args.target_dataset
SOURCE_VOCAB, TARGET_VOCAB = args.source_vocab, args.target_vocab
MAX_LEN_SOURCE = args.max_len_source
MAX_LEN_TARGET = args.max_len_target

BATCH_SIZE = args.batch_size    # input batch size for training
N_EPOCHS = args.epochs          # number of epochs to train
LR = args.lr                    # learning rate
NGPU = args.ngpu                # number of GPUs
PARALLEL = args.parallel        # use all GPUs

TOTAL_GPUs = torch.cuda.device_count() # Number of total GPUs available

if NGPU:
    assert TOTAL_GPUs >= NGPU, '{} GPUs not available! Only {} GPU(s) available'.format(NGPU, TOTAL_GPUs)

DEVICE = args.device if args.device else 'cuda' if torch.cuda.is_available() else 'cpu'
if args.device_id and 'cuda' in DEVICE:
    DEVICE_ID = args.device_id
    torch.cuda.set_device(DEVICE_ID)

def main():
    torch.set_num_threads(1) # Prevent error on KeyboardInterrupt with multiple GPUs

    make_dirs(PROJECT_DIR, [CHECKPOINTS_DIR, PLOTS_DIR, LOGGING_DIR]) # Create all required directories if not present
    setup_logging(PROJECT_DIR, LOGGING_DIR) # Setup configuration for logging

    global_vars = globals().copy()
    print_config(global_vars) # Print all global variables defined above
    global SOURCE_VOCAB, TARGET_VOCAB
    max_len_source, max_len_target = MAX_LEN_SOURCE, MAX_LEN_TARGET
    train_loader, SOURCE_VOCAB, TARGET_VOCAB, max_len_source, max_len_target = generate_dataloader(PROJECT_DIR, DATA_DIR, SOURCE_DATASET, TARGET_DATASET, 'train', \
                                       SOURCE_VOCAB, TARGET_VOCAB, BATCH_SIZE, max_len_source, max_len_target, args.force)

    #MAX_LEN_SOURCE, MAX_LEN_TARGET = max_len_source, max_len_target
    val_loader, _, _, _, _ = generate_dataloader(PROJECT_DIR, DATA_DIR, SOURCE_DATASET, TARGET_DATASET, 'dev', \
                                     SOURCE_VOCAB, TARGET_VOCAB, BATCH_SIZE, max_len_source, max_len_target, args.force)
    #import pdb;pdb.set_trace()
    start_epoch = 0 # Initialize starting epoch number (used later if checkpoint loaded)
    stop_epoch = N_EPOCHS+start_epoch # Store epoch upto which model is trained (used in case of KeyboardInterrupt)

    logging.info('Creating models...')
    #make sure encoder doesn't have less layers than decoder.
    encoder = RNNEncoder(kind = 'gru',
                vocab_size = SOURCE_VOCAB,
                embed_size = 256,
                hidden_size = 256,
                num_layers = 1,
                bidirectional = True,
                return_type='full_last_layer',
                prob_dropout_hidden=0)


    decoder = RNNDecoder(
            vocab_size = TARGET_VOCAB,
            embed_size = 256,
            fc_hidden_size=512,
            attn=None,
            encoder_directions=encoder.num_directions,
            encoder_hidden_size=encoder.hidden_size, \
            dropout=0.5,num_layers = 1)

    logging.info('Done.')

    # Define criteria and optimizer
    criterion_train = nn.CrossEntropyLoss()
    criterion_test = nn.CrossEntropyLoss(reduction='sum')
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=LR)

    train_loss_history, train_accuracy_history = [], []
    val_loss_history, val_accuracy_history = [], []

    # Load model state dicts if required
    if CHECKPOINT_FILE:
        encoder, decoder, optimizer, train_loss_history, val_loss_history, \
        train_accuracy_history, val_accuracy_history, epoch_trained = \
            load_checkpoint(encoder, decoder, optimizer, CHECKPOINT_FILE, PROJECT_DIR, CHECKPOINTS_DIR, DEVICE)
        start_epoch = epoch_trained # Start from (epoch_trained+1) if checkpoint loaded

    # Check if model is to be parallelized
    if TOTAL_GPUs > 1 and (PARALLEL or NGPU):
        DEVICE_IDs = range(TOTAL_GPUs) if PARALLEL else range(NGPU)
        logging.info('Using {} GPUs...'.format(len(DEVICE_IDs)))
        encoder = nn.DataParallel(encoder, device_ids=DEVICE_IDs)
        decoder = nn.DataParallel(decoder, device_ids=DEVICE_IDs)
        logging.info('Done.')
    encoder = encoder.to(DEVICE)
    decoder = decoder.to(DEVICE)

    early_stopping = EarlyStopping(mode='minimize', min_delta=0, patience=10)
    best_epoch = start_epoch+1

    for epoch in range(start_epoch+1, N_EPOCHS+start_epoch+1):
        try:
            train_losses = train(
                encoder=encoder,
                decoder=decoder,
                criterion=criterion_train,
                dataloader=train_loader,
                optimizer=optimizer,
                device=DEVICE,
                epoch=epoch,
                max_len_target=max_len_target
            )

            val_loss, val_pred, val_true = test(
                encoder=encoder,
                decoder=decoder,
                dataloader=val_loader,
                criterion=criterion_test,
                device=DEVICE
            )

            accuracy_train = accuracy(encoder, decoder, train_loader, criterion_test, DEVICE)
            accuracy_val = accuracy(encoder, decoder, val_loader, criterion_test, DEVICE)
            train_loss_history.extend(train_losses)
            val_loss_history.append(val_loss)
            train_accuracy_history.append(accuracy_train)
            val_accuracy_history.append(accuracy_val)

            logging.info('TRAIN Epoch: {}\tAverage loss: {:.4f}, Accuracy: {:.0f}%'.format(epoch, np.sum(train_losses), accuracy_train))
            logging.info('VAL   Epoch: {}\tAverage loss: {:.4f}, Accuracy: {:.0f}%\n'.format(epoch, val_loss, accuracy_val))

            if early_stopping.is_better(val_loss):
                logging.info('Saving current best model checkpoint...')
                save_checkpoint(encoder, decoder, optimizer, train_loss_history, val_loss_history, \
                            train_accuracy_history, val_accuracy_history, epoch, SOURCE_DATASET, TARGET_DATASET, \
                            PROJECT_DIR, CHECKPOINTS_DIR, PARALLEL or NGPU)
                logging.info('Done.')
                logging.info('Removing previous best model checkpoint...')
                remove_checkpoint(SOURCE_DATASET, TARGET_DATASET, PROJECT_DIR, CHECKPOINTS_DIR, best_epoch)
                logging.info('Done.')
                best_epoch = epoch

            if early_stopping.stop(val_loss) or round(accuracy_val) == 100:
                logging.info('Stopping early after {} epochs.'.format(epoch))
                stop_epoch = epoch
                break
        except KeyboardInterrupt:
            logging.info('Keyboard Interrupted!')
            stop_epoch = epoch-1
            break

    # Save the model checkpoints
    logging.info('Dumping model and results...')
    print_config(global_vars) # Print all global variables before saving checkpointing
    save_checkpoint(encoder, decoder, optimizer, train_loss_history, val_loss_history, \
                    train_accuracy_history, val_accuracy_history, stop_epoch, SOURCE_DATASET, \
                    TARGET_DATASET, PROJECT_DIR, CHECKPOINTS_DIR, PARALLEL or NGPU)
    logging.info('Done.')

    if len(train_loss_history) and len(val_loss_history):
        logging.info('Plotting and saving loss histories...')
        fig = plt.figure(figsize=(10,8))
        plt.plot(train_loss_history, alpha=0.5, color='blue', label='train')
        xticks = [epoch*len(train_loader) for epoch in range(1, len(val_loss_history)+1)]
        plt.plot(xticks, val_loss_history, alpha=0.5, color='orange', label='test')
        plt.legend()
        plt.title('Loss vs. Iterations')
        save_plot(PROJECT_DIR, PLOTS_DIR, fig, 'loss_vs_iterations.png')
        logging.info('Done.')

    if len(train_accuracy_history) and len(val_accuracy_history):
        logging.info('Plotting and saving accuracy histories...')
        fig = plt.figure(figsize=(10,8))
        plt.plot(train_accuracy_history, alpha=0.5, color='blue', label='train')
        plt.plot(val_accuracy_history, alpha=0.5, color='orange', label='test')
        plt.legend()
        plt.title('Accuracy vs. Iterations')
        save_plot(PROJECT_DIR, PLOTS_DIR, fig, 'accuracies_vs_iterations.png')
        logging.info('Done.')

if __name__ == '__main__':
    main()
