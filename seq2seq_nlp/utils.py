import numpy as np
import pandas as pd
import os
import logging
import time
import sys
import pickle
import heapq
from types import ModuleType
from pprint import pformat

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sacrebleu import corpus_bleu

PAD, PAD_IDX = '<pad>', 0
UNK, UNK_IDX = '<unk>', 1
SOS, SOS_IDX = '<sos>', 2
EOS, EOS_IDX = '<eos>', 3

def train_rnn(encoder, decoder, dataloader, criterion, optimizer, epoch, max_len_target, \
              clip_param, device, teacher_forcing_prob, joint_hidden_ec):
    loss_hist = []
    for batch_idx, (source, source_lens, target, target_lens) in enumerate(dataloader):
        source, source_lens  = source.to(device), source_lens.to(device)
        target, target_lens = target.to(device), target_lens.to(device)
        encoder.train()
        decoder.train()

        encoder_output, encoder_hidden = encoder(source, source_lens)
        # Doing complete teacher forcing first and then will add the probability based teacher forcing
        if joint_hidden_ec:
            decoder_hidden_step = decoder._init_state(source.size(0))
        else:
            if decoder.num_layers == 1:
                decoder_hidden_step = encoder_hidden[-1].unsqueeze(0)
            else:
                decoder_hidden_step = encoder_hidden

        input_seq = torch.tensor([SOS_IDX]*source.size(0)).to(device)
        loss = 0.
        # use teacher forcing is applied to an entire batch rather than over time steps
        use_teacher_forcing = np.random.rand() < teacher_forcing_prob

        max_batch_target_len = target_lens.data.max().item()
        if decoder.attn :
            prev_context_vec = torch.zeros((source.size(0), decoder.hidden_size)).to(device)
        else:
            prev_context_vec = None
        for step in range(max_batch_target_len):
            decoder_output_step, decoder_hidden_step, attn_weights_step, prev_context_vec = \
                    decoder(input_seq, decoder_hidden_step, source_lens,
                            encoder_output, context_vec=prev_context_vec)
            if use_teacher_forcing:
                input_seq = target[:, step]
            else:
                input_seq = decoder_output_step.topk(1, dim=1)[1].squeeze(1).detach()
            loss += criterion(decoder_output_step, target[:,step])

        # Take per-element average, subtracting the sos tokens for accurate computation
        loss /= (target_lens.data.sum().item())

        optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        nn.utils.clip_grad_norm_(encoder.parameters(), clip_param)
        nn.utils.clip_grad_norm_(decoder.parameters(), clip_param)

        optimizer.step()

        # Accurately compute loss, because of different batch size
        loss_train = loss.item() * len(source)/ len(dataloader.dataset)
        loss_hist.append(loss_train)

        # Print 25 times in a batch; if dataset too small print every time (to avoid division by 0)
        if (batch_idx+1) % max(1, (len(dataloader.dataset)//(25*source.shape[0]))) == 0:
            logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx+1) * source.shape[0], len(dataloader.dataset),
                100. * (batch_idx+1) / len(dataloader), loss.item()))

    optimizer.zero_grad()
    return loss_hist

# Different training loop for the case where we are using CNN encoder, assuming we will be applying this without attention
def train_cnn(encoder, decoder, dataloader, criterion, optimizer, epoch, max_len_target, \
              clip_param, device, teacher_forcing_prob, joint_hidden_ec):
    loss_hist = []
    for batch_idx, (source, source_lens, target, target_lens) in enumerate(dataloader):
        source, source_lens  = source.to(device), source_lens.to(device)
        target, target_lens = target.to(device), target_lens.to(device)
        encoder.train()
        decoder.train()

        encoder_output, encoder_hidden = encoder(source, source_lens)
        decoder_hidden_step = get_decoder_hidden_from_cnn(decoder.num_layers, encoder_hidden)

        input_seq = torch.tensor([SOS_IDX]*source.size(0)).to(device)
        loss = 0.
        # use teacher forcing is applied to an entire batch rather than over time steps
        use_teacher_forcing = np.random.rand() < teacher_forcing_prob

        max_batch_target_len = target_lens.data.max().item()

        for step in range(max_batch_target_len):
            decoder_output_step, decoder_hidden_step, _, _ = \
                    decoder(input_seq, decoder_hidden_step)

            if use_teacher_forcing:
                input_seq = target[:, step] # Change this line to change what to give as the next input to the decoder
            else:
                input_seq = decoder_output_step.topk(1, dim=1)[1].squeeze(1).detach()
            loss += criterion(decoder_output_step, target[:,step])

        # Take per-element average, subtracting the sos tokens for accurate computation
        loss /= (target_lens.data.sum().item())

        optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(encoder.parameters(), clip_param)
        nn.utils.clip_grad_norm_(decoder.parameters(), clip_param)

        optimizer.step()

        # Accurately compute loss, because of different batch size
        loss_train = loss.item() * len(source)/ len(dataloader.dataset)
        loss_hist.append(loss_train)

        # Print 25 times in a batch; if dataset too small print every time (to avoid division by 0)
        if (batch_idx+1) % max(1, (len(dataloader.dataset)//(25*source.shape[0]))) == 0:
            logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx+1) * source.shape[0], len(dataloader.dataset),
                100. * (batch_idx+1) / len(dataloader), loss.item()))

    optimizer.zero_grad()
    return loss_hist

# Works on batch size of 1 for the val loader
def test_rnn(encoder, decoder, dataloader, criterion, epoch, max_len_target, \
             id2token, token2id, device, joint_hidden_ec):
    loss_test = 0.
    encoder.eval()
    decoder.eval()
    all_output_sentences, all_target_sentences = [], []
    # Note that the batch size is just 1 here
    with torch.no_grad():
        for batch_idx, (source, source_len, target, target_len) in enumerate(dataloader):
            source, source_len  = source.to(device), source_len.to(device)
            target, target_len = target.to(device), target_len.to(device)
            # Init the decoder outputs with zeros and then fill them up with the values
            encoder_output, encoder_hidden = encoder(source, source_len)
            # Doing complete teacher forcing first and then will add the probability based teacher forcing
            if decoder.num_layers == 1:
                if encoder.num_directions == 1:
                    decoder_hidden_step = encoder_hidden[-1].unsqueeze(0)
                else:
                    decoder_hidden_step = encoder_hidden[-2:]
            else:
                decoder_hidden_step = encoder_hidden
            max_len_target = 2*source_len

            loss = 0.

            input_seq = torch.tensor([SOS_IDX]*source.size(0)).to(device)

            decoder_outputs = np.zeros((source.size(0), max_len_target))
            if decoder.attn :
                prev_context_vec = torch.zeros((source.size(0), decoder.hidden_size)).to(device)
            else:
                prev_context_vec = None
            for step in range(max_len_target):
                decoder_output_step, decoder_hidden_step, attn_weights_step, prev_context_vec = decoder(input_seq, decoder_hidden_step,
                 source_len, encoder_output, context_vec=prev_context_vec)

                input_seq = decoder_output_step.topk(1, dim=1)[1].squeeze(1).detach()
                if step < target_len:
                    loss += criterion(decoder_output_step, target[:,step])
                current_output = input_seq.cpu().numpy()
                if current_output == token2id[EOS]:
                    break
                decoder_outputs[:,step] = current_output

            target_sentences, output_sentences = get_all_sentences(target.cpu().numpy(), decoder_outputs, \
                                                                   id2token, token2id)

            all_output_sentences.extend(output_sentences)
            all_target_sentences.extend(target_sentences)
            loss /= (target_len.data.sum().item()) # Take per-element average

            # Accurately compute loss, because of different batch size
            loss_test += loss.item() * len(source)/ len(dataloader.dataset)

    rand_idx = np.random.randint(len(dataloader.dataset))
    logging.info('VAL  Epoch: {}\tTarget: {}\n'.format(epoch, all_target_sentences[rand_idx]))
    logging.info('VAL  Epoch: {}\tGreedy output: {}\n'.format(epoch, all_output_sentences[rand_idx]))

    bleu_score = corpus_bleu(all_output_sentences, [all_target_sentences], lowercase=True).score

    return loss_test, bleu_score

def test_cnn(encoder, decoder, dataloader, criterion, epoch, max_len_target, \
             id2token, token2id, device, joint_hidden_ec):
    loss_test = 0.
    encoder.eval()
    decoder.eval()
    all_output_sentences, all_target_sentences = [], []
    # Note that the batch size is just 1 here
    with torch.no_grad():
        for batch_idx, (source, source_len, target, target_len) in enumerate(dataloader):
            source, source_len  = source.to(device), source_len.to(device)
            target, target_len = target.to(device), target_len.to(device)
            # Init the decoder outputs with zeros and then fill them up with the values
            encoder_output, encoder_hidden = encoder(source, source_len)
            # Doing complete teacher forcing first and then will add the probability based teacher forcing

            decoder_hidden_step = get_decoder_hidden_from_cnn(decoder.num_layers, encoder_hidden)

            max_len_target = 2*source_len

            loss = 0.

            input_seq = torch.tensor([SOS_IDX]*source.size(0)).to(device)

            decoder_outputs = np.zeros((source.size(0), max_len_target))

            for step in range(max_len_target):
                decoder_output_step, decoder_hidden_step, _, _= decoder(input_seq, decoder_hidden_step)

                input_seq = decoder_output_step.topk(1, dim=1)[1].squeeze(1).detach()
                if step < target_len:
                    loss += criterion(decoder_output_step, target[:,step])
                current_output = input_seq.cpu().numpy()
                if current_output == token2id[EOS]:
                    break
                decoder_outputs[:,step] = current_output

            target_sentences, output_sentences = get_all_sentences(target.cpu().numpy(), decoder_outputs, \
                                                                   id2token, token2id)

            all_output_sentences.extend(output_sentences)
            all_target_sentences.extend(target_sentences)
            loss /= (target_len.data.sum().item()) # Take per-element average

            # Accurately compute loss, because of different batch size
            loss_test += loss.item() * len(source)/ len(dataloader.dataset)

    rand_idx = np.random.randint(len(dataloader.dataset))
    logging.info('VAL  Epoch: {}\tTarget: {}\n'.format(epoch, all_target_sentences[rand_idx]))
    logging.info('VAL  Epoch: {}\tGreedy output: {}\n'.format(epoch, all_output_sentences[rand_idx]))

    bleu_score = corpus_bleu(all_output_sentences, [all_target_sentences], lowercase=True).score

    return loss_test, bleu_score

def test_beam_search(encoder, decoder, dataloader, criterion, epoch,
                     max_len_target, id2token, token2id, device, beam_size):
    encoder.eval()
    decoder.eval()
    all_output_sentences, all_target_sentences = [], []
    with torch.no_grad():
        start = time.time()
        for batch_idx, (source, source_lens, target, target_lens) in enumerate(dataloader):
            source, source_lens  = source.to(device), source_lens.to(device)
            target, target_lens = target.to(device), target_lens.to(device)

            batch_size = source.size(0)
            # Init the decoder outputs with zeros and then fill them up with the values
            encoder_output, encoder_hidden = encoder(source, source_lens)
            # Doing complete teacher forcing first and then will add the probability based teacher forcing
            if decoder.num_layers == 1:
                if encoder.num_directions == 1:
                    decoder_hidden = encoder_hidden[-1].unsqueeze(0)
                else:
                    decoder_hidden = encoder_hidden[-2:]
            else:
                decoder_hidden = encoder_hidden

            input_seqs = torch.tensor([SOS_IDX]*source.size(0)).to(device)

            max_batch_target_len = target_lens.data.max().item()
            # max_len_target = 2*source_lens
            batch_beam = [
                {'prob': [],
                'seq_ixs': [0 for _ in range(max_len_target)],
                'item': []}
                for i in range(batch_size)
            ]

            if decoder.attn :
                prev_context_vec = torch.zeros((source.size(0), decoder.hidden_size)).to(device)
            else:
                prev_context_vec = None

            # first step
            decoder_output, decoder_hidden, attn_weights_step, prev_context_vec = decoder(input_seqs, decoder_hidden, \
                 source_lens, encoder_output, context_vec=prev_context_vec)

            decoder_output, decoder_output_ixs = decoder_output.topk(beam_size, dim=1)
            input_seqs = decoder_output_ixs
            decoder_output = decoder_output.cpu().tolist()
            decoder_output_ixs = decoder_output_ixs.cpu().tolist()

            for i in range(batch_size):
                batch_beam[i]['prob'] = decoder_output[i]
                batch_beam[i]['seq_ixs'] = \
                    [[decoder_output_ixs[i][k]] for k in range(beam_size)]
                batch_beam[i]['item'] = [(-decoder_output[i][k], decoder_output_ixs[i][k], -1, k, decoder_output_ixs[i][k] == token2id['<eos>']) for k in range(beam_size)]

            hidden_states_beam = [decoder_hidden for l in range(beam_size)]
            for step in range(max_len_target):
                # (batch_size, beam_size * beam_size)
                decoder_outputs_beam = [[] for i in range(batch_size)]
                new_hidden_states = []
                for k in range(beam_size):
                    hidden_state = hidden_states_beam[k]
                    dec_out_beam, dec_hidden_beam, attn_weights_beam, prev_context_vec = decoder(input_seqs[:,k], hidden_state, \
                    source_lens, encoder_output, context_vec=prev_context_vec)

                    new_hidden_states.append(dec_hidden_beam)
                    dec_out_vals, dec_out_ixs = (
                        dec_out_beam.topk(beam_size, dim=1))
                    dec_out_vals = dec_out_vals.cpu().numpy()
                    dec_out_ixs = dec_out_ixs.cpu().numpy() # (B, beam_size)
                    for i in range(batch_size):
                        prev_item = batch_beam[i]['item'][k]
                        is_done = prev_item[4]
                        if is_done:
                            # from a finished branch only add the final state of branch
                            if not isinstance(prev_item, tuple) or len(prev_item) != 5:
                                print("WRONG PREV_ITEM: ", prev_item)
                                print("Type: ", type(prev_item))
                            heapq.heappush(decoder_outputs_beam[i], prev_item)
                        else:
                            # from an unfinished branch add the "beam_size" most probable words to come after
                            prev_prob = batch_beam[i]['prob'][k]
                            # store negative probabilities because heapq returns smallest values
                            new_probs = - dec_out_vals[i]
                            new_ixs = dec_out_ixs[i]
                            new_is_done = [ix == token2id['<eos>'] for ix in new_ixs]
                            new_outputs = tuple(zip(new_probs, new_ixs, [k] * beam_size, new_is_done))
                            for item in new_outputs:
                                if not isinstance(item, tuple) or len(item) != 4:
                                    print("WRONG ITEM: ", item)
                                heapq.heappush(decoder_outputs_beam[i], item)

                batch_is_done = True
                input_seqs = torch.zeros(batch_size, beam_size, dtype=torch.long)
                for i in range(batch_size):
                    best_K = heapq.nsmallest(beam_size,
                                             decoder_outputs_beam[i])
                    if not isinstance(best_K[0], tuple):
                        print("WRONG BEST K: ", best_K[0])
                    decoder_outputs_beam[i] = []
                    batch_beam_dict = {
                        'prob': [],
                        'seq_ixs': [],
                        'item': [],
                    }
                    for j, item in enumerate(best_K):
                        input_seqs[i, j] = int(item[1])
                        is_old_item = len(item) == 5
                        if is_old_item:
                            neg_prob, word_ix, prev_beam_ix, beam_ix, is_done = item
                            batch_beam_dict['prob'].append(- neg_prob)
                            sequence_ixs = batch_beam[i]['seq_ixs'][beam_ix]
                            batch_beam_dict['seq_ixs'].append(sequence_ixs)
                            item = (neg_prob, word_ix, beam_ix, j, is_done)
                            batch_beam_dict['item'].append(item)
                        else:
                            neg_prob, word_ix, prev_beam_ix, is_done = item
                            batch_is_done = False
                            batch_beam_dict['prob'].append(- neg_prob)
                            sequence_ixs = batch_beam[i]['seq_ixs'][prev_beam_ix] + [word_ix]
                            batch_beam_dict['seq_ixs'].append(sequence_ixs)
                            item = (neg_prob, word_ix, prev_beam_ix, j, is_done)
                            batch_beam_dict['item'].append(item)
                        if isinstance(decoder.rnn, nn.LSTM):
                            decoder_hidden = (new_hidden_states[prev_beam_ix][0][:, i, :], \
                                              new_hidden_states[prev_beam_ix][1][:, i, :])
                            hidden_states_beam[j][0][:, i, :] = decoder_hidden[0]
                            hidden_states_beam[j][1][:, i, :] = decoder_hidden[1]
                        else:
                            decoder_hidden = new_hidden_states[prev_beam_ix][:, i, :]
                            hidden_states_beam[j][:, i, :] = decoder_hidden
                    batch_beam[i] = batch_beam_dict

                input_seqs = input_seqs.to(device)

                if batch_is_done:
                    break

                if step < max_batch_target_len:
                    target_step = target[:, step]

            final_outputs = np.zeros((batch_size, 2*max_len_target))
            avg = []
            for i in range(batch_size):
                best_ix = 0
                best_seq = batch_beam[i]["seq_ixs"][best_ix]
                final_outputs[i, :len(best_seq)] = best_seq

            target_sentences, output_sentences = get_all_sentences(target.cpu().numpy(), final_outputs, \
                                                                   id2token, token2id)
            all_output_sentences.extend(output_sentences)
            all_target_sentences.extend(target_sentences)

        rand_idx = np.random.randint(len(dataloader.dataset))
        logging.info('VAL  Epoch: {}\tAvg time of loop with beam search: {:.3f}\n'.format(epoch, (time.time()-start) / len(dataloader)))
        logging.info('VAL  Epoch: {}\tTarget: {}\n'.format(epoch, all_target_sentences[rand_idx]))
        logging.info('VAL  Epoch: {}\tbeam output: {}\n'.format(epoch, all_output_sentences[rand_idx]))

    bleu_score = corpus_bleu(all_output_sentences, [all_target_sentences]).score
    return bleu_score

def test_gold(encoder, decoder, dataloader, criterion, epoch, max_len_target, \
             id2token, token2id, device, joint_hidden_ec,source_dataset,project_dir, data_dir):
    encoder.eval()
    decoder.eval()
    all_output_sentences, all_target_sentences = [], []
    batch_idx_to_select = np.random.choice(range(len(dataloader)),10,replace=False)
    all_weights_selected_sentences = []
    selected_output_sentences = []
    selected_target_sentences = []
    selected_source_sentences = []

    with torch.no_grad():
        for batch_idx, (source, source_len, source_orig_sent, target_orig_sent) in enumerate(dataloader):
            source, source_len  = source.to(device), source_len.to(device)

            encoder_output, encoder_hidden = encoder(source, source_len)
            current_weights_all_steps = None
            if(batch_idx in batch_idx_to_select):
                current_weights_all_steps = []
            if decoder.num_layers == 1:
                if encoder.num_directions == 1:
                    decoder_hidden_step = encoder_hidden[-1].unsqueeze(0)
                else:
                    decoder_hidden_step = encoder_hidden[-2:]
            else:
                decoder_hidden_step = encoder_hidden
            max_len_target = 2*source_len
            input_seq = torch.tensor([SOS_IDX]*source.size(0)).to(device)
            decoder_outputs = np.zeros((source.size(0), max_len_target))

            if decoder.attn :
                prev_context_vec = torch.zeros((source.size(0), decoder.hidden_size)).to(device)
            else:
                prev_context_vec = None

            for step in range(max_len_target):
                decoder_output_step, decoder_hidden_step, attn_weights_step, prev_context_vec = decoder(input_seq, decoder_hidden_step,
                 source_len, encoder_output, context_vec=prev_context_vec)
                if current_weights_all_steps is not None:
                    current_weights_all_steps.append(attn_weights_step.squeeze(1).cpu().numpy())
                input_seq = decoder_output_step.topk(1, dim=1)[1].squeeze(1).detach()
                current_output = input_seq.cpu().numpy()
                if current_output == token2id[EOS]:
                    break
                decoder_outputs[:,step] = current_output
            output_sentence = convert_idxs_to_sentence(decoder_outputs[0], id2token, token2id)
            all_output_sentences.append(output_sentence)
            all_target_sentences.append(target_orig_sent[0])

            if(batch_idx in batch_idx_to_select):
                selected_output_sentences.append(output_sentence)
                selected_source_sentences.append(source_orig_sent[0])
                selected_target_sentences.append(target_orig_sent[0])
                all_weights_selected_sentences.append(current_weights_all_steps)

    bleu_score = corpus_bleu(all_output_sentences, [all_target_sentences], lowercase=True).score

    logging.info(f'Test Bleu score for the dataset:{source_dataset} is {bleu_score}')

    save_object(all_output_sentences,os.path.join(project_dir, data_dir, f'all_output_sentences.{source_dataset}'))
    save_object(all_target_sentences,os.path.join(project_dir, data_dir, f'all_target_sentences.{source_dataset}'))
    save_object(all_weights_selected_sentences,os.path.join(project_dir, data_dir, f'all_weights_selected_sentences.{source_dataset}'))
    save_object(selected_output_sentences,os.path.join(project_dir, data_dir, f'selected_output_sentences.{source_dataset}'))
    save_object(selected_target_sentences,os.path.join(project_dir, data_dir, f'selected_target_sentences.{source_dataset}'))
    save_object(selected_source_sentences,os.path.join(project_dir, data_dir, f'selected_source_sentences.{source_dataset}'))
    return bleu_score

def get_decoder_hidden_from_cnn(decoder_num_layers, encoder_hidden):
    decoder_hidden_step = []
    for layer in range(decoder_num_layers):
        decoder_hidden_step.append(encoder_hidden)
    return torch.stack(decoder_hidden_step, dim=0)

def convert_idxs_to_sentence(idxs, id2token, token2id):
    tokens = [id2token[idxs[i]] for i in range(len(idxs)) if idxs[i] not in \
              [token2id['<pad>'], token2id['<sos>'], token2id['<eos>']]]
    return ' '.join(tokens)

def get_all_sentences(target, output, id2token, token2id):
    all_sentences_target, all_sentences_output = [], []
    for i in range(len(target)):
        target_sentence = convert_idxs_to_sentence(target[i], id2token, token2id)
        output_sentence = convert_idxs_to_sentence(output[i], id2token, token2id)
        all_sentences_target.append(target_sentence)
        all_sentences_output.append(output_sentence)

    return all_sentences_target, all_sentences_output

def print_config(vars_dict):
    vars_dict = {key: value for key, value in vars_dict.items() if key == key.upper() \
                 and not isinstance(value, ModuleType)}
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
        data = [f(line) for line in fin]
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
                    train_bleu_history, val_greedy_bleu_history, val_beam_bleu_history, epoch, args, project_dir,
                    checkpoints_dir, is_parallel=False):
    state_dict = {
        'encoder_state_dict': encoder.module.state_dict() if is_parallel else encoder.state_dict(),
        'decoder_state_dict': decoder.module.state_dict() if is_parallel else decoder.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'train_loss_history': train_loss_history,
        'val_loss_history': val_loss_history,
        'train_bleu_history': train_bleu_history,
        'val_greedy_bleu_history': val_greedy_bleu_history,
        'val_beam_bleu_history': val_beam_bleu_history
    }

    params = [args.source_dataset, args.target_dataset, args.source_vocab, \
              args.target_vocab, args.max_len_source, args.max_len_target, \
              args.encoder_type, args.num_directions, args.encoder_num_layers, \
              args.decoder_type,args.decoder_num_layers, args.encoder_emb_size, args.decoder_emb_size, \
              args.encoder_hid_size, \
              args.decoder_hid_size,int(args.use_attn)]

    state_dict_name = 'state_dict' + '_{}'*len(params) + '_epoch{}.pkl'
    state_dict_name = state_dict_name.format(*params, epoch)
    state_dict_path = os.path.join(project_dir, checkpoints_dir, state_dict_name)
    logging.info('Saving checkpoint "{}"...'.format(state_dict_path))
    torch.save(state_dict, state_dict_path)
    logging.info('Done.')

def remove_checkpoint(args, project_dir, checkpoints_dir, epoch):
    params = [args.source_dataset, args.target_dataset, args.source_vocab, \
              args.target_vocab, args.max_len_source, args.max_len_target, \
              args.encoder_type, args.num_directions, args.encoder_num_layers, \
              args.decoder_type,args.decoder_num_layers, args.encoder_emb_size, args.decoder_emb_size, \
              args.encoder_hid_size, \
              args.decoder_hid_size,int(args.use_attn)]

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
    train_bleu_history, val_greedy_bleu_history, val_beam_bleu_history = [], [], []
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
        val_greedy_bleu_history = state_dict['val_greedy_bleu_history']
        val_beam_bleu_history = state_dict['val_beam_bleu_history']

        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        logging.info('Successfully loaded checkpoint.')

    else:
        raise FileNotFoundError('No checkpoint found at "{}"!'.format(state_dict_path))

    return encoder, decoder, optimizer, train_loss_history, val_loss_history, \
            train_bleu_history, val_greedy_bleu_history, val_beam_bleu_history, epoch_trained

def save_model(model, model_name, epoch, args, project_dir, checkpoints_dir):
    params = [args.source_dataset, args.target_dataset, args.source_vocab, \
              args.target_vocab, args.max_len_source, args.max_len_target, \
              args.encoder_type, args.num_directions, args.encoder_num_layers, \
              args.decoder_type,args.decoder_num_layers, args.encoder_emb_size, args.decoder_emb_size, \
              args.encoder_hid_size, \
              args.decoder_hid_size,int(args.use_attn)]

    checkpoint_name = model_name + '_{}'*len(params) + '_epoch{}.pt'
    checkpoint_name = checkpoint_name.format(*params, epoch)
    checkpoint_path = os.path.join(project_dir, checkpoints_dir, checkpoint_name)
    logging.info('Saving checkpoint "{}"...'.format(checkpoint_path))
    torch.save(model.to('cpu'), checkpoint_path)
    logging.info('Done.')

def load_model(project_dir, checkpoints_dir, checkpoint_file):
    checkpoint_path = os.path.join(project_dir, checkpoints_dir, checkpoint_file)
    if os.path.exists(checkpoint_path):
        logging.info('Loading checkpoint "{}"...'.format(checkpoint_path))
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
