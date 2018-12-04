import os
import logging
from torch.utils.data import DataLoader
from collections import Counter
from seq2seq_nlp.datasets import *
from seq2seq_nlp.utils import *
from functools import partial
# not removing punctuation: does BLEU count correctly translated punctuation?
# not tokenizing

PAD, PAD_IDX = '<pad>', 0
UNK, UNK_IDX = '<unk>', 1
SOS, SOS_IDX = '<sos>', 2
EOS, EOS_IDX = '<eos>', 3

def add_special_symbols(data):
    logging.info("Adding 'SOS' and 'EOS' symbols to target data")
    return [[SOS] + row + [EOS] for row in data]

def get_counts(project_dir, data_dir, dataset, vocab_size, is_source, force=False):
    logging.info(f'Getting counts for {dataset}')
    counts_path = os.path.join(project_dir, data_dir, f'counts.{dataset}.p')
    have_counts = os.path.isfile(counts_path)
    if not force and have_counts:
        logging.info('Already have counts')
        counts = load_object(counts_path)
    else:
        logging.info(f'Getting all tokens for {dataset}')
        all_tokens_path = os.path.join(project_dir, data_dir, f'all_tokens.{dataset}.p')
        have_all_tokens = os.path.isfile(all_tokens_path)
        if not force and have_all_tokens:
            logging.info('Already have all tokens')
            all_tokens = load_object(all_tokens_path)
        else:
            data_path = os.path.join(project_dir, data_dir, f'train.tok.clean.{dataset}')
            data = load_raw_data(data_path)
            logging.info(f'Is source: {is_source}')
            all_tokens = [tok for sentence in data for tok in sentence]
            save_object(all_tokens, all_tokens_path)
            if not is_source:
                data = add_special_symbols(data)
        logging.info(f'Number of tokens: {len(all_tokens)}')
        counts = Counter(all_tokens)
        logging.info('Saving counts')
        save_object(counts, counts_path)
    top_counts = counts.most_common(vocab_size)
    top_counts = [(word, count) for word, count in top_counts if count >= 5]
    return top_counts

def get_vocab(project_dir, data_dir, dataset, id2token_path, token2id_path, \
              vocab_size=50000, is_source=True, force=False):
    have_vocab = os.path.isfile(id2token_path)
    logging.info(f'Building vocabulary for {dataset}')
    if force:
        logging.info('Saving vocabulary')
        top_counts = get_counts(project_dir, data_dir, dataset, vocab_size, is_source, force)
        vocabulary, counts = zip(*top_counts)
        if is_source:
            id2token = [PAD, UNK] + list(vocabulary)
        else:
            id2token = [PAD, UNK, SOS, EOS] + list(vocabulary)
        id2token = dict(zip(range(len(id2token)), id2token))
        token2id = {tok: ix for ix, tok in id2token.items()}
        save_object(id2token, id2token_path)
        save_object(token2id, token2id_path)
    elif have_vocab:
        logging.info('Already have vocabulary')
        id2token = load_object(id2token_path)
        token2id = load_object(token2id_path)
    logging.info(f'Vocab size for {dataset} data: {len(id2token)}')
    return id2token, token2id

def get_data_indices(project_dir, data_dir, kind, dataset, vocab_size, is_source, \
                     id2token=None, token2id=None, force=False):
    def transform(data, token2id):
        data_indices = [[token2id[tok] if tok in token2id else token2id[UNK] \
                         for tok in sentence] for sentence in data]
        return data_indices
    logging.info(f'Building {kind} data of indices for {dataset}')
    data_ind_path = os.path.join(project_dir, data_dir, f'{kind}.tok.ind.{vocab_size}.{dataset}')
    id2token_path = os.path.join(project_dir, data_dir, f'id2token.{vocab_size}.{dataset}.p')
    token2id_path = os.path.join(project_dir, data_dir, f'token2id.{vocab_size}.{dataset}.p')
    have_indices = os.path.isfile(data_ind_path) and os.path.isfile(id2token_path) and os.path.isfile(token2id_path)
    if kind == 'train':
        if force:
            data = load_raw_data(os.path.join(project_dir, data_dir, f'train.tok.clean.{dataset}'))
            id2token, token2id = get_vocab(project_dir, data_dir, dataset, id2token_path, \
                                           token2id_path, vocab_size, is_source, force)
            data_ind = transform(data, token2id)
            logging.info('Saving the indices data')
            dump_ind_data(data_ind, data_ind_path)
        elif have_indices:
            logging.info(f'Already have the indices data')
            data_ind = load_ind_data(data_ind_path)
            id2token = load_object(id2token_path)
            token2id = load_object(token2id_path)
    else:
        data = load_raw_data(os.path.join(project_dir, data_dir, f'{kind}.tok.clean.{dataset}'))
        data_ind = transform(data, token2id)
        logging.info('Saving the indices data')
        dump_ind_data(data_ind, data_ind_path)

    # Check the dictionary by loading random token from it
    random_ind = np.random.randint(0, len(id2token)-1)
    random_tok = id2token[random_ind]
    assert random_ind == token2id[random_tok] and random_tok == id2token[random_ind]

    return data_ind, id2token, token2id

def generate_dataloader(project_dir, data_dir, source_dataset, target_dataset, kind, \
                        source_vocab_size, target_vocab_size, batch_size, max_len_source, \
                        max_len_target, id2token=None, token2id=None, force=False):
    '''
    kind (str): possible values - 'train' or 'dev' or 'test'
    '''
    datasets = [source_dataset, target_dataset]
    vocab_sizes = [source_vocab_size, target_vocab_size]
    are_source = [True, False]

    data = {}
    if kind == 'train':
        assert id2token is None and token2id is None
        id2token, token2id = {}, {}
    else:
        assert id2token is not None and token2id is not None


    source_path = os.path.join(project_dir, data_dir, f'{kind}.tok.{source_dataset}')
    target_path = os.path.join(project_dir, data_dir, f'{kind}.tok.{target_dataset}')
    source_clean_path = os.path.join(project_dir, data_dir, f'{kind}.tok.clean.{source_dataset}')
    target_clean_path = os.path.join(project_dir, data_dir, f'{kind}.tok.clean.{target_dataset}')

    clean_paired_files(source_path, target_path, source_clean_path, target_clean_path)

    for is_source, dataset, vocab_size in zip(are_source, datasets, vocab_sizes):
        key = 'source' if is_source else 'target'
        if kind == 'train':
            data[key], id2token[key], token2id[key] = \
                get_data_indices(project_dir, data_dir, kind, dataset, \
                                 vocab_size, is_source, None, None, force)
        else:
            data[key], _, _ = \
                get_data_indices(project_dir, data_dir, kind, dataset, \
                                 vocab_size, is_source, id2token[key], \
                                 token2id[key], force)

    logging.info('Creating Dataset')
    if(max_len_source == -1):
        max_len_source = np.array([len(sent) for sent in data['source']], dtype=np.int64).max()
    if(max_len_target == -1):
        max_len_target = np.array([len(sent) for sent in data['target']], dtype=np.int64).max()

    logging.info('Creating Dataloader')
    dataset = NMTDataset(data, max_len_source=max_len_source, max_len_target=max_len_target, pad_idx=PAD_IDX)
    shuffle = True if kind == 'train' else False
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, \
                            collate_fn=partial(nmt_collate_fn, max_len_source=max_len_source, \
                            max_len_target = max_len_target))

    if kind == 'train':
        return dataloader, len(id2token['source']), len(id2token['target']), \
               max_len_source, max_len_target, id2token, token2id
    return dataloader
