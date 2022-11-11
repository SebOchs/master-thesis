import datasets
from datasets import Dataset
from models import *
import numpy as np
import os
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import sys
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Slurm fix
sys.path.append(os.getcwd())


def training(model_folder, batch_size=8, lr=2e-5, epochs=16, split=1, norm=False, occ_or_ohv=''):
    """
    Training script
    :param model_folder: string / determines approach
    :param batch_size: int
    :param lr: float
    :param epochs: int
    :param split: float / percentage of training data to use
    :param norm: boolean / if rules should be normalized
    :param occ_or_ohv: string / use occurence vectors or one hot encodings
    :return: None
    """
    def preprocess(example):
        # ensures max length of 128 tokens per text
        return tokenizer(example['text'], padding='max_length', truncation=True, max_length=128)

    # Load data
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    train = Dataset.from_pandas(pd.read_csv("dataset/bne/train.csv")).remove_columns('Unnamed: 0')
    dev = Dataset.from_pandas(pd.read_csv("dataset/bne/dev.csv")).remove_columns('Unnamed: 0')

    # Add rule info to data when indicated
    if occ_or_ohv == 'occ':
        train_rules = np.load("dataset/bne/rule_occ_vectorstrain.npy", allow_pickle=True)
        dev_rules = np.load("dataset/bne/rule_occ_vectorsdev.npy", allow_pickle=True)
        train = train.add_column('rules', list(train_rules))
        dev = dev.add_column('rules', list(dev_rules))
        model_folder += '_occ'
    if occ_or_ohv == 'ohv':
        train_rules = np.load("dataset/bne/one_hot_vectorstrain.npy", allow_pickle=True)
        dev_rules = np.load("dataset/bne/one_hot_vectorsdev.npy", allow_pickle=True)
        train = train.add_column('rules', list(train_rules))
        dev = dev.add_column('rules', list(dev_rules))
        model_folder += '_ohv'

    if norm:
        model_folder += '_norm'

    # determine percentage of train data to use
    if split < 1:
        train = train.shuffle(123).select(range(int(len(train)*split)))
        model_folder += '_' + str(int(split*100))

    # Preprocess data
    train = train.map(preprocess).remove_columns('text')
    dev = dev.map(preprocess).remove_columns('text')

    # Create dataloaders
    train = train.with_format('torch')
    dev = dev.with_format('torch')

    train_loader = DataLoader(train, batch_size=batch_size)
    val_loader = DataLoader(dev, batch_size=batch_size)

    # Train with different seeds
    for seed in [1234 #, 5678, 9012, 3456, 7890 ]:
                ]:
        pl.seed_everything(seed)
        # Checkpointing
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join('models', model_folder, str(seed)),
            monitor='mf1',
            mode="max",
            filename='{epoch}-{mf1:.4f}'
        )
        # Early Stopping
        early = EarlyStopping(
            monitor='mf1',
            mode="max",
            patience=4,
            verbose=False
        )

        # Initialize model and trainer
        if model_folder.startswith('baseline'):
            model = BaselineBERT(batch_size=batch_size, lr=lr)
        if model_folder.startswith('lin'):
            model = LinLayerExtension(batch_size=batch_size, lr=lr, norm=norm)
        if model_folder.startswith('add'):
            model = AddLayer(batch_size=batch_size, lr=lr, norm=norm)
        if model_folder.startswith('kenn'):
            model = KennLayer(batch_size=batch_size, lr=lr, norm=norm)

        trainer = pl.Trainer(
            accelerator='gpu',
            devices=1,
            max_epochs=epochs,
            callbacks=[checkpoint_callback, early],
            num_sanity_val_steps=0
        )

        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    # Baseline
    training('baseline')
    training('baseline', split=0.5)
    training('baseline', split=0.1)
    # lin layer approaches
    training('linlayer', occ_or_ohv='occ')
    training('linlayer', occ_or_ohv='occ', norm=True)
    training('linlayer', occ_or_ohv='occ', split=0.5)
    training('linlayer', occ_or_ohv='occ', norm=True, split=0.5)
    training('linlayer', occ_or_ohv='occ', split=0.1)
    training('linlayer', occ_or_ohv='occ', norm=True, split=0.1)
    training('linlayer', occ_or_ohv='ohv')
    training('linlayer', occ_or_ohv='ohv', norm=True)
    training('linlayer', occ_or_ohv='ohv', split=0.5)
    training('linlayer', occ_or_ohv='ohv', norm=True, split=0.5)
    training('linlayer', occ_or_ohv='ohv', split=0.1)
    training('linlayer', occ_or_ohv='ohv', norm=True, split=0.1)
    # add layer approaches
    training('addlayer', occ_or_ohv='occ')
    training('addlayer', occ_or_ohv='occ', norm=True)
    training('addlayer', occ_or_ohv='occ', split=0.5)
    training('addlayer', occ_or_ohv='occ', norm=True, split=0.5)
    training('addlayer', occ_or_ohv='occ', split=0.1)
    training('addlayer', occ_or_ohv='occ', norm=True, split=0.1)
    training('addlayer', occ_or_ohv='ohv')
    training('addlayer', occ_or_ohv='ohv', norm=True)
    training('addlayer', occ_or_ohv='ohv', split=0.5)
    training('addlayer', occ_or_ohv='ohv', norm=True, split=0.5)
    training('addlayer', occ_or_ohv='ohv', split=0.1)
    training('addlayer', occ_or_ohv='ohv', norm=True, split=0.1)
    # kenn layer approaches
    training('kennlayer', occ_or_ohv='occ')
    training('kennlayer', occ_or_ohv='occ', norm=True)
    training('kennlayer', occ_or_ohv='occ', split=0.5)
    training('kennlayer', occ_or_ohv='occ', norm=True, split=0.5)
    training('kennlayer', occ_or_ohv='occ', split=0.1)
    training('kennlayer', occ_or_ohv='occ', norm=True, split=0.1)
    training('kennlayer', occ_or_ohv='ohv')
    training('kennlayer', occ_or_ohv='ohv', norm=True)
    training('kennlayer', occ_or_ohv='ohv', split=0.5)
    training('kennlayer', occ_or_ohv='ohv', norm=True, split=0.5)
    training('kennlayer', occ_or_ohv='ohv', split=0.1)
    training('kennlayer', occ_or_ohv='ohv', norm=True, split=0.1)

