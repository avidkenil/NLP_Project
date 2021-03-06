import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-dir', metavar='PROJECT_DIR', dest='project_dir', \
                        help='path to project directory', required=False, default='.')
    parser.add_argument('--data-dir', metavar='DATA_DIR', dest='data_dir', \
                        help='path to data directory', required=False, default='data')
    parser.add_argument('--source-dataset', metavar='SOURCE_DATASET', dest='source_dataset', \
                        help='name of source dataset file in data directory', required=False, default='vi')
    parser.add_argument('--target-dataset', metavar='TARGET_DATASET', dest='target_dataset', \
                        help='name of target dataset file in data directory', required=False, default='en')
    parser.add_argument('--checkpoints-dir', metavar='CHECKPOINTS_DIR', dest='checkpoints_dir', \
                        help='path to checkpoints directory', required=False, default='checkpoints')
    parser.add_argument('--load-ckpt', metavar='LOAD_CHECKPOINT', dest='load_ckpt', \
                        help='name of checkpoint file to load', required=False)
    parser.add_argument('--load-enc-ckpt', metavar='LOAD_ENC_CKPT', dest='load_enc_ckpt', \
                        help='name of encoder model to load', required=False)
    parser.add_argument('--load-dec-ckpt', metavar='LOAD_DEC_CKPT', dest='load_dec_ckpt', \
                        help='name of decoder model to load', required=False)
    parser.add_argument('--source-vocab', metavar='SOURCE_VOCAB', dest='source_vocab', \
                        help='source dataset vocabulary size', required=False, default=50000, type=int)
    parser.add_argument('--target-vocab', metavar='TARGET_VOCAB', dest='target_vocab', \
                        help='target dataset vocabulary size', required=False, default=50000, type=int)
    parser.add_argument('--max-len-source', metavar='MAX_LEN_SOURCE', dest='max_len_source', \
                        help='max sentence length of source', required=False, type=int, default=100)
    parser.add_argument('--max-len-target', metavar='MAX_LEN_TARGET', dest='max_len_target', \
                        help='max sentence length of target', required=False, type=int, default=100)
    parser.add_argument('--unk-threshold', metavar='UNK_THRESHOLD', dest='unk_threshold', help='threshold '\
                        'of count below which words are to be treated as UNK. Please use force if you change this.', required=False, type=int, default=5)
    parser.add_argument('--batch-size', metavar='BATCH_SIZE', dest='batch_size', help='batch size', \
                        required=False, type=int, default=32)
    parser.add_argument('--epochs', metavar='EPOCHS', dest='epochs', help='number of epochs', \
                        required=False, type=int, default=10)
    parser.add_argument('--device', metavar='DEVICE', dest='device', help='device', required=False)
    parser.add_argument('--device-id', metavar='DEVICE_ID', dest='device_id', help='device id of gpu', \
                        required=False, type=int)
    parser.add_argument('--ngpu', metavar='NGPU', dest='ngpu', help='number of GPUs to use (0,1,...,ngpu-1)', \
                        required=False, type=int)
    parser.add_argument('--parallel', action='store_true', help='use all GPUs available', required=False)
    parser.add_argument('--lr', metavar='LR', dest='lr', help='learning rate', required=False, \
                        type=float, default=1e-3)
    parser.add_argument('--encoder-type', metavar='ENCODER_TYPE', dest='encoder_type', help='type of encoder model '\
                        '(gru | lstm | rnn | cnn), default="gru"', required=False, default='gru')
    parser.add_argument('--decoder-type', metavar='DECODER_TYPE', dest='decoder_type', help='type of decoder model '\
                        '(gru | lstm), default="gru"', required=False, default='gru')
    parser.add_argument('--num-directions', metavar='NUM_DIRECTIONS', dest='num_directions', help='number of directions '\
                        'in encoder, default=2', required=False, type=int, default=2)
    parser.add_argument('--encoder-num-layers', metavar='ENCODER_NUM_LAYERS', dest='encoder_num_layers', help='number of '\
                        'layers in encoder, default=2', required=False, type=int, default=2)
    parser.add_argument('--decoder-num-layers', metavar='DECODER_NUM_LAYERS', dest='decoder_num_layers', help='number of '\
                        'layers in decoder, default=2', required=False, type=int, default=2)
    parser.add_argument('--encoder-emb-size', metavar='ENCODER_EMB_SIZE', dest='encoder_emb_size', help='embedding size '\
                        'of encoder, default=256', required=False, type=int, default=256)
    parser.add_argument('--decoder-emb-size', metavar='DECODER_EMB_SIZE', dest='decoder_emb_size', help='embedding size '\
                        'of decoder, default=256', required=False, type=int, default=256)
    parser.add_argument('--encoder-hid-size', metavar='ENCODER_HID_SIZE', dest='encoder_hid_size', help='hidden size '\
                        'of encoder, default=256', required=False, type=int, default=256)
    parser.add_argument('--encoder-dropout', metavar='ENCODER_DROPOUT', dest='encoder_dropout', \
                        help='dropout rate in encoder, default=0.2', required=False, type=float, default=0.2)
    parser.add_argument('--decoder-dropout', metavar='DECODER_DROPOUT', dest='decoder_dropout', \
                        help='dropout rate in decoder FC layer, default=0.1', required=False, type=float, default=0.1)
    parser.add_argument('--clip-param', metavar='CLIP_PARAM', dest='clip_param', \
                        help='clip parameter value for exploding gradients', required=False, type=float, default=5.0)
    parser.add_argument('--beam-size', metavar='BEAM_SIZE', dest='beam_size', \
                        help='Beam size to use during beam search', required=False, type=int, default=5)
    parser.add_argument('--force', action='store_true', help='overwrites all existing dumped data sets')
    parser.add_argument('--teacher-forcing-prob', metavar='TEACHER_FORCING_PROB', dest='teacher_forcing_prob', \
                        help='clip parameter value for exploding gradients', required=False, type=float, default=0.5)
    parser.add_argument('--use-attn', action='store_true', help='Use attention in the encoder decoder architecture')
    args = parser.parse_args()

    return args
