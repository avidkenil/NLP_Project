# Natural Language Processing with Representation Learning

## Machine Translation for Vietnamese → English and Chinese → English

Members:
  - Mihir Rana
  - Kenil Tanna
  - Diogo Mesquita
  - Yassine Kadiri

## Requirements
For ease of setup, we have created a [requirements.yaml](https://github.com/avidkenil/NLP_Project/blob/master/requirements.yaml) file which will create a conda environment with the name `nlp_project` and install all dependencies and requirements into that environment. To do this:
  - Install Anaconda and run:
```
conda env create -f requirements.yaml
```
  - Optionally, if you want to run it on a GPU, install CUDA and cuDNN

## Installation
Again, for simplicity, we have created a module with the name `seq2seq-nlp` which can be installed directly into pip by running the following command from the main project directory:
```
pip install -e .
```

## Usage
```
usage: main.py [-h] [--project-dir PROJECT_DIR] [--data-dir DATA_DIR]
               [--source-dataset SOURCE_DATASET]
               [--target-dataset TARGET_DATASET]
               [--checkpoints-dir CHECKPOINTS_DIR]
               [--load-ckpt LOAD_CHECKPOINT] [--load-enc-ckpt LOAD_ENC_CKPT]
               [--load-dec-ckpt LOAD_DEC_CKPT] [--source-vocab SOURCE_VOCAB]
               [--target-vocab TARGET_VOCAB] [--max-len-source MAX_LEN_SOURCE]
               [--max-len-target MAX_LEN_TARGET] [--batch-size BATCH_SIZE]
               [--epochs EPOCHS] [--device DEVICE] [--device-id DEVICE_ID]
               [--ngpu NGPU] [--parallel] [--lr LR] [--encoder ENCODER]
               [--num-directions NUM_DIRECTIONS]
               [--encoder-num-layers ENCODER_NUM_LAYERS]
               [--decoder-num-layers DECODER_NUM_LAYERS]
               [--encoder-emb-size ENCODER_EMB_SIZE]
               [--decoder-emb-size DECODER_EMB_SIZE]
               [--encoder-hid-size ENCODER_HID_SIZE]
               [--decoder-hid-size DECODER_HID_SIZE]
               [--encoder-dropout ENCODER_DROPOUT]
               [--decoder-dropout DECODER_DROPOUT] [--clip-param CLIP_PARAM]
               [--force]

optional arguments:
  -h, --help                                  show this help message and exit
  --project-dir PROJECT_DIR                   path to project directory
  --source-dataset SOURCE_DATASET             name of source dataset file in data directory, default='vi'
  --target-dataset TARGET_DATASET             name of target dataset file in data directory, default='en'
  --data-dir DATA_DIR                         path to data directory, default="data/vi-en"
  --checkpoints-dir CHECKPOINTS_DIR           path to checkpoints directory
  --load-ckpt LOAD_CHECKPOINT                 name of checkpoint file to load
    --load-enc-ckpt LOAD_ENC_CKPT             name of encoder file to load
  --load-dec-ckpt LOAD_DEC_CKPT               name of decoder file to load
  --source-vocab SOURCE_VOCAB                 source dataset vocabulary size, default=50000
  --target-vocab TARGET_VOCAB                 target dataset vocabulary size, default=50000
  --max-len-source MAX_LEN_SOURCE             max sentence length of source, default=300
  --max-len-target MAX_LEN_TARGET             max sentence length of target, default=600
  --batch-size BATCH_SIZE                     batch size, default=32
  --epochs EPOCHS                             number of epochs, default=10
  --device DEVICE                             cuda | cpu, default=cuda
                                              device to train on
  --device-id DEVICE_ID                       device id of gpu, default=0
  --ngpu NGPU                                 number of GPUs to use (0,1,...,ngpu-1)
  --parallel                                  use all GPUs available
  --lr LR                                     learning rate, default=1e-4
  --encoder ENCODER                           type of encoder model (gru | lstm), default="gru"
  --num-directions NUM_DIRECTIONS             number of directions in encoder, default=2
  --encoder-num-layers ENCODER_NUM_LAYERS     number of layers in encoder, default=1
  --decoder-num-layers DECODER_NUM_LAYERS     number of layers in decoder, default=1
  --encoder-emb-size ENCODER_EMB_SIZE         embedding size of encoder, default=256
  --decoder-emb-size DECODER_EMB_SIZE         embedding size of decoder, default=256
  --encoder-hid-size ENCODER_HID_SIZE         hidden size of encoder, default=256
  --decoder-hid-size DECODER_HID_SIZE         hidden size of decoder, default=512
  --encoder-dropout ENCODER_DROPOUT           dropout rate in encoder, default=0.
  --decoder-dropout DECODER_DROPOUT           dropout rate in decoder FC layer, default=0.
  --clip-param CLIP_PARAM                     clip parameter value for exploding gradients
  --force                                     overwrites all existing dumped data sets
```
