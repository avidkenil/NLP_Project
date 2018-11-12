# DS-GA 1011 Natural Language Processing with Representation Learning

# Neural Machine Translation for Vietnamese (Vi) → English (En) and Chinese (Zh) → En

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
usage: main.py [-h] [--project-dir PROJECT_DIR]
               [--source-dataset SOURCE_DATASET]
               [--target-dataset TARGET_DATASET] [--data-dir DATA_DIR]
               [--checkpoints-dir CHECKPOINTS_DIR]
               [--load-ckpt LOAD_CHECKPOINT] [--batch-size BATCH_SIZE]
               [--epochs EPOCHS] [--device DEVICE] [--device-id DEVICE_ID]
               [--ngpu NGPU] [--parallel] [--lr LR] [--force]

optional arguments:
  -h, --help            				show this help message and exit
  --project-dir PROJECT_DIR				path to project directory
  --source-dataset SOURCE_DATASET			name of source dataset file in data directory
  --target-dataset TARGET_DATASET			name of target dataset file in data directory
  --data-dir DATA_DIR   				path to data directory (used if different from "data")
  --checkpoints-dir CHECKPOINTS_DIR			path to checkpoints directory
  --load-ckpt LOAD_CHECKPOINT				name of checkpoint file to load
  --batch-size BATCH_SIZE				batch size, default=64
  --epochs EPOCHS       				number of epochs, default=10
  --device DEVICE       				cuda | cpu, default=cuda
							device to train on
  --device-id DEVICE_ID					device id of gpu, default=0
  --ngpu NGPU           				number of GPUs to use (0,1,...,ngpu-1)
  --parallel            				use all GPUs available
  --lr LR               				learning rate
  --force               				overwrites all existing dumped data sets
```
