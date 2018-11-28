import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-dir', metavar='PROJECT_DIR', dest='project_dir', \
                        help='path to project directory', required=False, default='.')
    parser.add_argument('--data-dir', metavar='DATA_DIR', dest='data_dir', \
                        help='path to data directory (used if different from "data")', \
                        required=False, default='data/vi-en')
    parser.add_argument('--source-dataset', metavar='SOURCE_DATASET', dest='source_dataset', \
                        help='name of source dataset file in data directory', required=False, default='vi')
    parser.add_argument('--target-dataset', metavar='TARGET_DATASET', dest='target_dataset', \
                        help='name of target dataset file in data directory', required=False, default='en')
    parser.add_argument('--checkpoints-dir', metavar='CHECKPOINTS_DIR', dest='checkpoints_dir', \
                        help='path to checkpoints directory', required=False, default='checkpoints')
    parser.add_argument('--source-vocab', metavar='SOURCE_VOCAB', dest='source_vocab', \
                        help='source dataset vocabulary size', required=False, default=50000, type=int)
    parser.add_argument('--target-vocab', metavar='TARGET_VOCAB', dest='target_vocab', \
                        help='target dataset vocabulary size', required=False, default=50000, type=int)
    parser.add_argument('--load-ckpt', metavar='LOAD_CHECKPOINT', dest='load_ckpt', \
                        help='name of checkpoint file to load', required=False)
    parser.add_argument('--max-len', metavar='MAX_LEN', dest='max_len', help='max sentence length', \
                        required=False, type=int, default=300)
    parser.add_argument('--batch-size', metavar='BATCH_SIZE', dest='batch_size', help='batch size', \
                        required=False, type=int, default=64)
    parser.add_argument('--epochs', metavar='EPOCHS', dest='epochs', help='number of epochs', \
                        required=False, type=int, default=10)
    parser.add_argument('--device', metavar='DEVICE', dest='device', help='device', required=False)
    parser.add_argument('--device-id', metavar='DEVICE_ID', dest='device_id', help='device id of gpu', \
                        required=False, type=int)
    parser.add_argument('--ngpu', metavar='NGPU', dest='ngpu', help='number of GPUs to use (0,1,...,ngpu-1)', \
                        required=False, type=int)
    parser.add_argument('--parallel', action='store_true', help='use all GPUs available', required=False)
    parser.add_argument('--lr', metavar='LR', dest='lr', help='learning rate', required=False, \
                        type=float, default=1e-4)
    parser.add_argument('--force', action='store_true', help='overwrites all existing dumped data sets')
    args = parser.parse_args()

    return args
