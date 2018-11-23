import sys
import os.path
import pickle as pkl
import logging
import argparse
from pprint import pprint
from collections import Counter
import utils

# not removing punctuation: does blue count correctly translated punctuation?
# not tokenizing

PAD = '<pad>'
UNK = '<unk>'
SOS = '<sos>'
EOS = '<eos>'

def add_special_symbols_target(data):
    logging.info("Adding 'SOS' and 'EOS' symbol to data")
    return [[SOS] + row + [EOS] for row in data]

def get_counts(dirname, suffix, vocab_size, is_source):
    logging.info(f"getting counts for {suffix}")
    counts_path = os.path.join(dirname, f'counts.{suffix}.p')
    have_counts = os.path.isfile(counts_path)
    if have_counts:
        logging.info("already have counts")
        counts = utils.load_object(counts_path)
    else:
        logging.info(f"getting all tokens for {suffix}")
        all_tokens_path = os.path.join(dirname, f'all_tokens.{suffix}.p')
        have_all_tokens = os.path.isfile(all_tokens_path)
        if have_all_tokens:
            logging.info("already have all tokens")
            all_tokens = utils.load_object(all_tokens_path)
        else:
            data_path = os.path.join(dirname, f'train.tok.{suffix}')
            data = utils.load_raw_data(data_path)
            logging.info(f"Is source: {is_source}")
            if not is_source:
                data = add_special_symbols_target(data)
            all_tokens = [tok for sentence in data for tok in sentence]
            utils.save_object(all_tokens, all_tokens_path)
        logging.info(f"number of tokens: {len(all_tokens)}")
        counts = Counter(all_tokens)
        logging.info("saving counts")
        utils.save_object(counts, counts_path)
    return counts.most_common(vocab_size)


def get_vocab(dirname, suffix, vocab_size=50000, is_source=True):
    ind2tok_path = os.path.join(dirname, f'ind2tok.{vocab_size}.{suffix}.p')
    tok2ind_path = os.path.join(dirname, f'tok2ind.{vocab_size}.{suffix}.p')
    have_vocab = os.path.isfile(ind2tok_path)
    logging.info(f"building vocabulary of size {vocab_size + 2} for {suffix}")
    if have_vocab:
        logging.info("already have vocabulary")
        ind2tok = utils.load_object(ind2tok_path)
        tok2ind = utils.load_object(tok2ind_path)
    else:
        ind2tok = [PAD, UNK]
        top_counts = get_counts(dirname, suffix, vocab_size, is_source)
        ind2tok += [tok for tok, _ in top_counts]
        tok2ind = {tok: ix for ix, tok in enumerate(ind2tok)}
        utils.save_object(ind2tok, ind2tok_path)
        utils.save_object(tok2ind, tok2ind_path)
        logging.info("saving vocabulary")
    return ind2tok, tok2ind


def get_data_indices(kind, dirname, suffix, vocab_size, is_source):
    '''
    kind (str): possible values - 'train' or 'dev' or 'test'
    '''
    def transform(data, tok2ind):
        data_indices = [[tok2ind[tok] if tok in tok2ind else tok2ind[UNK]
                         for tok in sentence] for sentence in data]
        return data_indices
    logging.info(f"building {kind} data of indices for {suffix}")
    data_ind_path = os.path.join(dirname,
                                      f'{kind}.tok.ind.{vocab_size}.{suffix}')
    have_indices = os.path.isfile(data_ind_path)
    if have_indices:
        logging.info(f"already have the indices data")
        data_ind = utils.load_txt(data_ind_path)
    else:
        data = utils.load_raw_data(os.path.join(dirname, f'{kind}.tok.{suffix}'))
        _, tok2ind = get_vocab(dirname, suffix, vocab_size, is_source)
        data_ind = transform(data, tok2ind)
        logging.info("saving the indices data")
        utils.dump_ind_data(data_ind, data_ind_path)
    return data_ind


def main(source_name, target_name, dirname,
         source_vocab_size, target_vocab_size):
    suffixes = [source_name, target_name]
    vocab_sizes = [source_vocab_size, target_vocab_size]
    are_source = [True, False]
    path = dirname

    train_ind_data = []
    val_ind_data = []
    test_ind_data = []

    for is_source, suffix, vocab_size in zip(are_source, suffixes, vocab_sizes):
        train_ind_data.append(get_data_indices('train', path, suffix,
                                               vocab_size, is_source))
        val_ind_data.append(get_data_indices('dev', path, suffix,
                                             vocab_size, is_source))
        test_ind_data.append(get_data_indices('test', path, suffix,
                                              vocab_size, is_source))


if __name__ == '__main__':
    source_name = sys.argv[1]
    target_name = sys.argv[2]
    dirname = sys.argv[3]
    source_vocab_size = int(sys.argv[4])
    target_vocab_size = int(sys.argv[5])
    log_path = sys.argv[6]
    logging.basicConfig(filename=log_path, filemode='w', level=logging.INFO,
                        format='%(asctime)s : %(levelname)s : %(message)s')
    main(source_name, target_name, dirname,
         source_vocab_size, target_vocab_size)

    # print("Test")
    # path = '../data/vi-en'
    # suffix = 'en'
    # is_source = False
    # vocab_size = 10000

    # counts = get_counts(path, suffix, vocab_size, is_source)
    # counts_set = {word for word, count in counts}
    # print(f"Counts of {suffix}")
    # pprint(counts[:10], indent=2)
    # print(f"Hello in counts? {'Hello' in counts_set}")
    # print(f"Ca in counts? {'Ca' in counts_set}")
    # print(f"EOS in counts? {EOS in counts_set}")



