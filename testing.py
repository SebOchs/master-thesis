import datasets
from datasets import Dataset
from models import *
import numpy as np
import os
import pandas as pd
import pytorch_lightning as pl
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, accuracy_score
import torch
from torch.utils.data import DataLoader
import tqdm
from transformers import AutoTokenizer

device = torch.device("cuda")


def test_model(mode, occ_or_ohv=False):
    """

    :param mode:
    :return:
    """

    def preds_to_let(preds, labs):
        """
        translates BNE levels (0, 1, 2, 3, 4, 5, 6) to LET (A2, B1, B2, (C))
        :param preds:
        :param labs:
        :return:
        """
        label_dict = {0: [0, 1, 2, 3], 1: [3, 4], 2: [4, 5], 3: [6]}
        new_predictions = []
        for x, y in zip(preds, labs):
            if x in label_dict[y]:
                new_predictions.append(y)
            elif x in [0, 1, 2]:
                new_predictions.append(0)
            elif x in [5]:
                new_predictions.append(2)
            elif x in [6]:
                new_predictions.append(3)
            else:
                new_predictions.append(1)
        return new_predictions

    def preprocess(example):
        return tokenizer(example['text'], padding='max_length', truncation=True, max_length=128)

    # Load test data
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    test = Dataset.from_pandas(pd.read_csv("dataset/bne/test.csv")).remove_columns('Unnamed: 0')
    test_ut = Dataset.from_pandas(pd.read_csv("dataset/bne/test_ut.csv")).remove_columns('Unnamed: 0')
    test_let = Dataset.from_pandas(pd.read_csv("dataset/bne/test_let.csv")).remove_columns('Unnamed: 0')
    translation = {'A2': 0, 'B1': 1, 'B2': 2}
    new_labels = list(map(lambda x: translation[x], test_let['labels']))
    test_let = test_let.remove_columns('labels').add_column('labels', new_labels)
    # Add rule info to data
    if occ_or_ohv == 'occ':
        test_rules = np.load("dataset/bne/rule_occ_vectorstest.npy", allow_pickle=True)
        test_ut_rules = np.load("dataset/bne/rule_occ_vectorstest_ut.npy", allow_pickle=True)
        test_let_rules = np.load("dataset/bne/rule_occ_vectorstest_let.npy", allow_pickle=True)

    if occ_or_ohv == 'ohv':
        test_rules = np.load("dataset/bne/one_hot_vectorstest.npy", allow_pickle=True)
        test_ut_rules = np.load("dataset/bne/one_hot_vectorstest_ut.npy", allow_pickle=True)
        test_let_rules = np.load("dataset/bne/one_hot_vectorstest_let.npy", allow_pickle=True)

    if occ_or_ohv:
        test = test.add_column('rules', list(test_rules))
        test_ut = test_ut.add_column('rules', list(test_ut_rules))
        test_let = test_let.add_column('rules', list(test_let_rules))

    # Preprocess data
    test = test.map(preprocess).remove_columns('text')
    test_ut = test_ut.map(preprocess).remove_columns('text')
    test_let = test_let.map(preprocess).remove_columns('text')

    # Create dataloaders
    test = test.with_format('torch')
    test_ut = test_ut.with_format('torch')
    test_let = test_let.with_format('torch')

    test_loader = DataLoader(test, batch_size=16)
    test_ut_loader = DataLoader(test_ut, batch_size=16)
    test_let_loader = DataLoader(test_let, batch_size=8)

    for seed in os.listdir(os.path.join('models', mode)):
        # Load model corresponding to seed and mode
        path = os.path.join('models', mode, seed)
        if mode.startswith('baseline'):
            ckpt = BaselineBERT.load_from_checkpoint(os.path.join(path, os.listdir(path)[0]))
        elif mode.startswith('lin'):
            ckpt = LinLayerExtension.load_from_checkpoint(os.path.join(path, os.listdir(path)[0]))
        elif mode.startswith('add'):
            ckpt = AddLayer.load_from_checkpoint(os.path.join(path, os.listdir(path)[0]))
        elif mode.startswith('kenn'):
            ckpt = KennLayer.load_from_checkpoint(os.path.join(path, os.listdir(path)[0]))
        else:
            raise NotImplementedError('Either not implemented yet or check string mode.')

        trainer = pl.Trainer(accelerator='gpu', gpus=1)
        trainer.test(ckpt, dataloaders=test_loader)
        np.save(os.path.join(path, 'test_results.npy'), ckpt.test_results)
        trainer.test(ckpt, dataloaders=test_ut_loader)
        np.save(os.path.join(path, 'test_ut_results.npy'), ckpt.test_results)
        trainer.test(ckpt, dataloaders=test_let_loader)
        test_let_results = ckpt.test_results
        new_labs = test_let['labels'].tolist()
        new_preds = preds_to_let(ckpt.test_results['predictions'], test_let['labels'].tolist())
        test_let_results['acc'] = accuracy_score(new_labs, new_preds)
        test_let_results['mf1'] = f1_score(new_labs, new_preds, average='macro')
        np.save(os.path.join(path, 'test_let_results.npy'), test_let_results)


# test_model('baseline')
# test_model('linlayerext', occ_or_ohv='occ')
# test_model('addlayer', occ_or_ohv='occ')
test_model('kennlayer', occ_or_ohv='occ')
