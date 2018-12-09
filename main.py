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
from seq2seq_nlp.datasets import *


args = get_args()


# Globals
PROJECT_DIR = args.project_dir
SOURCE_DATASET, TARGET_DATASET = args.source_dataset, args.target_dataset
DATA_DIR,  PLOTS_DIR, LOGGING_DIR = args.data_dir, 'plots', 'logs-lstm'
args.data_dir = DATA_DIR = os.path.join(DATA_DIR, '{}-{}'\
                                        .format(SOURCE_DATASET, TARGET_DATASET))
CHECKPOINTS_DIR, CHECKPOINT_FILE = args.checkpoints_dir, args.load_ckpt
ENCODER_MODEL_CKPT, DECODER_MODEL_CKPT = args.load_enc_ckpt, args.load_dec_ckpt
SOURCE_VOCAB, TARGET_VOCAB = args.source_vocab, args.target_vocab
MAX_LEN_SOURCE, MAX_LEN_TARGET = args.max_len_source, args.max_len_target
UNK_THRESHOLD = args.unk_threshold
CLIP_PARAM = args.clip_param
BEAM_SIZE = args.beam_size

# Model hyperparameters
ENCODER_TYPE, DECODER_TYPE = args.encoder_type, args.decoder_type        # Type of encoder
NUM_DIRECTIONS = args.num_directions
assert NUM_DIRECTIONS in [1, 2]
BIDIRECTIONAL = True if NUM_DIRECTIONS == 2 else False
ENCODER_NUM_LAYERS, DECODER_NUM_LAYERS = args.encoder_num_layers, args.decoder_num_layers
# Make sure encoder doesn't have lesser layers than decoder
assert ENCODER_NUM_LAYERS >= DECODER_NUM_LAYERS
ENCODER_EMB_SIZE, DECODER_EMB_SIZE = args.encoder_emb_size, args.decoder_emb_size
ENCODER_HID_SIZE = args.encoder_hid_size
args.decoder_hid_size = DECODER_HID_SIZE = ENCODER_HID_SIZE*NUM_DIRECTIONS
ENCODER_DROPOUT, DECODER_DROPOUT = args.encoder_dropout, args.decoder_dropout
JOINT_HIDDEN_EC = False
TEACHER_FORCING_PROB = args.teacher_forcing_prob

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

    global SOURCE_VOCAB, TARGET_VOCAB, MAX_LEN_SOURCE, MAX_LEN_TARGET

    train_loader, SOURCE_VOCAB, TARGET_VOCAB, MAX_LEN_SOURCE, MAX_LEN_TARGET, id2token, token2id = \
        generate_dataloader(PROJECT_DIR, DATA_DIR, SOURCE_DATASET, TARGET_DATASET, 'train', SOURCE_VOCAB, \
                            TARGET_VOCAB, BATCH_SIZE, MAX_LEN_SOURCE, MAX_LEN_TARGET, UNK_THRESHOLD, None, \
                            None, args.force)

    # Print all global variables defined above (and updated vocabulary sizes / max sentence lengths)
    args.source_vocab, args.target_vocab = SOURCE_VOCAB, TARGET_VOCAB
    args.max_len_source, args.max_len_target = MAX_LEN_SOURCE, MAX_LEN_TARGET
    global_vars = globals().copy()
    print_config(global_vars)

    val_loader = generate_dataloader(PROJECT_DIR, DATA_DIR, SOURCE_DATASET, TARGET_DATASET, 'dev', \
                                     SOURCE_VOCAB, TARGET_VOCAB, BATCH_SIZE, MAX_LEN_SOURCE, MAX_LEN_TARGET, \
                                     UNK_THRESHOLD, id2token, token2id, args.force, nmt_collate_fn_train)
    #create data loader for greedy with batch size of 1 and give it the val collate function
    val_loader_greedy = generate_dataloader(PROJECT_DIR, DATA_DIR, SOURCE_DATASET, TARGET_DATASET, 'dev', \
                                     SOURCE_VOCAB, TARGET_VOCAB, 1, MAX_LEN_SOURCE, MAX_LEN_TARGET, \
                                     UNK_THRESHOLD, id2token, token2id, args.force, nmt_collate_fn_val)

    start_epoch = 0 # Initialize starting epoch number (used later if checkpoint loaded)
    stop_epoch = N_EPOCHS+start_epoch # Store epoch upto which model is trained (used in case of KeyboardInterrupt)

    logging.info('Creating models...')
    encoder = RNNEncoder(kind=ENCODER_TYPE,
                vocab_size=SOURCE_VOCAB,
                embed_size=ENCODER_EMB_SIZE,
                hidden_size=ENCODER_HID_SIZE,
                num_layers=ENCODER_NUM_LAYERS,
                bidirectional=BIDIRECTIONAL,
                dropout=ENCODER_DROPOUT,
                device=DEVICE)

    decoder = RNNDecoder(
            vocab_size=TARGET_VOCAB,
            embed_size=DECODER_EMB_SIZE,
            kind=DECODER_TYPE,
            encoder_directions=NUM_DIRECTIONS,
            encoder_hidden_size=encoder.hidden_size,
            num_layers=DECODER_NUM_LAYERS,
            fc_hidden_size=DECODER_HID_SIZE,
            attn=None,
            dropout=DECODER_DROPOUT,
            joint_hidden_ec = JOINT_HIDDEN_EC,
            device=DEVICE)
    logging.info('Done.')

    logging.info(encoder)
    logging.info(decoder)
    # Define criteria and optimizer
    # Ignore padding indexes
    criterion_train = nn.NLLLoss(reduction='sum', ignore_index=0)
    criterion_test = nn.NLLLoss(reduction='sum', ignore_index=0)
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=LR)

    train_loss_history, train_bleu_history = [], []
    val_loss_history, val_greedy_bleu_history, val_beam_bleu_history = [], [], []

    # Load model state dicts / models if required
    epoch_trained = 0
    if CHECKPOINT_FILE: # First check for state dicts
        encoder, decoder, optimizer, train_loss_history, val_loss_history, \
        train_bleu_history, val_greedy_bleu_history, val_beam_bleu_history, epoch_trained = \
            load_checkpoint(encoder, decoder, optimizer, CHECKPOINT_FILE, \
                            PROJECT_DIR, CHECKPOINTS_DIR, DEVICE)
    elif ENCODER_MODEL_CKPT or DECODER_MODEL_CKPT: # Otherwise check for entire model
        if ENCODER_MODEL_CKPT:
            encoder, epoch_trained = load_model(PROJECT_DIR, CHECKPOINTS_DIR, ENCODER_MODEL_CKPT)
        if DECODER_MODEL_CKPT:
            decoder, epoch_trained_dec = load_model(PROJECT_DIR, CHECKPOINTS_DIR, DECODER_MODEL_CKPT)
            assert epoch_trained == epoch_trained_dec, \
                'Mismatch in epochs trained for encoder (={}) and decoder (={}).'\
                .format(epoch_trained, epoch_trained_dec)
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

    early_stopping = EarlyStopping(mode='maximize', min_delta=0, patience=100)
    best_epoch = start_epoch+1

    for epoch in range(start_epoch+1, N_EPOCHS+start_epoch+1):
        try:
            train_losses = train(
                encoder=encoder,
                decoder=decoder,
                criterion=criterion_train,
                dataloader=train_loader,
                optimizer=optimizer,
                epoch=epoch,
                max_len_target=MAX_LEN_TARGET,
                clip_param=CLIP_PARAM,
                device=DEVICE,
                teacher_forcing_prob = TEACHER_FORCING_PROB,
                joint_hidden_ec = JOINT_HIDDEN_EC
            )

            val_loss, val_greedy_bleu = test(
                encoder=encoder,
                decoder=decoder,
                dataloader=val_loader,
                criterion=criterion_test,
                epoch=epoch,
                max_len_target=MAX_LEN_TARGET,
                id2token=id2token['target'],
                token2id=token2id['target'],
                device=DEVICE,
                joint_hidden_ec = JOINT_HIDDEN_EC
            )

            # val_beam_bleu = test_beam_search(
            #     encoder=encoder,
            #     decoder=decoder,
            #     dataloader=val_loader,
            #     criterion=criterion_test,
            #     epoch=epoch,
            #     max_len_target=MAX_LEN_TARGET,
            #     id2token=id2token['target'],
            #     token2id=token2id['target'],
            #     device=DEVICE,
            #     beam_size=BEAM_SIZE
            # )
            val_beam_bleu = 0


            train_loss_history.extend(train_losses)
            val_loss_history.append(val_loss)
            val_greedy_bleu_history.append(val_greedy_bleu)
            val_beam_bleu_history.append(val_beam_bleu)

            logging.info('TRAIN Epoch: {}\tAverage loss: {:.4f}\n'.format(epoch, np.sum(train_losses)))
            logging.info('VAL   Epoch: {}\tAverage loss: {:.4f}, greedy BLEU: {:.4f}, beam BLEU: {:.4f}\n'.format(epoch, val_loss, val_greedy_bleu, val_beam_bleu))

            if early_stopping.is_better(val_loss):
                logging.info('Saving current best model checkpoint...')
                save_checkpoint(encoder, decoder, optimizer, train_loss_history, val_loss_history, \
                                train_bleu_history, val_greedy_bleu_history, val_beam_bleu_history, epoch, args, \
                                PROJECT_DIR, CHECKPOINTS_DIR, PARALLEL or NGPU)
                logging.info('Done.')
                logging.info('Removing previous best model checkpoint...')
                remove_checkpoint(args, PROJECT_DIR, CHECKPOINTS_DIR, best_epoch)
                logging.info('Done.')
                best_epoch = epoch

            if early_stopping.stop(val_beam_bleu):
                logging.info('Stopping early after {} epochs.'.format(epoch))
                stop_epoch = epoch
                break
        except KeyboardInterrupt:
            logging.info('Keyboard Interrupted!')
            stop_epoch = epoch - 1
            break

    # Save the model checkpoints
    logging.info('Dumping model and results...')
    print_config(global_vars) # Print all global variables before saving checkpointing
    save_checkpoint(encoder, decoder, optimizer, train_loss_history, val_loss_history, \
                    train_bleu_history, val_greedy_bleu_history, val_beam_bleu_history, stop_epoch, args, \
                    PROJECT_DIR, CHECKPOINTS_DIR, PARALLEL or NGPU)
    save_model(encoder, 'encoder', stop_epoch, args, PROJECT_DIR, CHECKPOINTS_DIR)
    save_model(decoder, 'decoder', stop_epoch, args, PROJECT_DIR, CHECKPOINTS_DIR)
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

    if len(train_bleu_history) and len(val_greedy_bleu_history)\
        and len(val_beam_bleu_history):
        logging.info('Plotting and saving BLEU histories...')
        fig = plt.figure(figsize=(10,8))
        plt.plot(train_bleu_history, alpha=0.5, color='blue', label='train')
        plt.plot(val_greedy_bleu_history, alpha=0.5, color='orange', label='test_greedy')
        plt.plot(val_beam_bleu_history, alpha=0.5, color='green', label=f'test_beam_{BEAM_SIZE}')
        plt.legend()
        plt.title('BLEU vs. Iterations')
        save_plot(PROJECT_DIR, PLOTS_DIR, fig, 'bleu_vs_iterations.png')
        logging.info('Done.')

if __name__ == '__main__':
    main()
