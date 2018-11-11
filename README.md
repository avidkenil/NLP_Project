# NLP_Project
Machine translation project for the DS:GA-1011 NLP class.


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
