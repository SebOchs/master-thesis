from collections import defaultdict
import datasets
import ltn
import numpy as np
import os
import pandas as pd
import random
from sklearn.metrics import f1_score, accuracy_score
import torch
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import transformers

# set seed
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)

# set gpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# copied from
# https://github.com/tommasocarraro/LTNtorch/blob/main/examples/2-multi_class_single_label_classification.ipynb
class LogitsToPredicate(torch.nn.Module):
    """
    This model has inside a logits model, that is a model which compute logits for the classes given an input example x.
    The idea of this model is to keep logits and probabilities separated. The logits model returns the logits for an example,
    while this model returns the probabilities given the logits model.

    In particular, it takes as input an example x and a class label l. It applies the logits model to x to get the logits.
    Then, it applies a softmax function to get the probabilities per classes. Finally, it returns only the probability related
    to the given class l.
    """

    def __init__(self, logits_model):
        super(LogitsToPredicate, self).__init__()
        self.logits_model = logits_model
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x, l):
        logits = self.logits_model(input_ids=x[:, :, 0], attention_mask=x[:, :, 1], token_type_ids=x[:, :, 2]).logits
        probs = self.softmax(logits)
        if torch.all(l == l[0]):
            out = probs[:, l[0]]
        else:
            raise ValueError("Labels are supposed to be the same")
        return out


def batch_encode_label(batch, label, keys):
    return torch.stack(list(transformers.BatchEncoding({k: v[batch['labels'] == label] for k, v in batch.items() if k in
                                                        keys}).values()), axis=-1)


def preprocess(example):
    # tokenize text
    return tokenizer(example['text'].tolist(), padding='max_length', truncation=True, max_length=256)


def axioms(batch, training=False):
    # model input
    x_0 = ltn.Variable('x_0', batch_encode_label(batch, 0, ['input_ids', 'attention_mask', 'token_type_ids']))
    x_1 = ltn.Variable('x_1', batch_encode_label(batch, 1, ['input_ids', 'attention_mask', 'token_type_ids']))
    x_2 = ltn.Variable('x_2', batch_encode_label(batch, 2, ['input_ids', 'attention_mask', 'token_type_ids']))

    # CTTR
    cttr_0 = ltn.Variable('cttr_0', batch_encode_label(batch, 0, ['CTTR']))
    cttr_1 = ltn.Variable('cttr_1', batch_encode_label(batch, 1, ['CTTR']))
    cttr_2 = ltn.Variable('cttr_2', batch_encode_label(batch, 2, ['CTTR']))

    # nbXP
    nbxp_0 = ltn.Variable('nbxp_0', batch_encode_label(batch, 0, ['nbXP']))
    nbxp_1 = ltn.Variable('nbxp_1', batch_encode_label(batch, 1, ['nbXP']))
    nbxp_2 = ltn.Variable('nbxp_2', batch_encode_label(batch, 2, ['nbXP']))

    # Spache
    spache_0 = ltn.Variable('spache_0', batch_encode_label(batch, 0, ['Spache']))
    spache_1 = ltn.Variable('spache_1', batch_encode_label(batch, 1, ['Spache']))
    spache_2 = ltn.Variable('spache_2', batch_encode_label(batch, 2, ['Spache']))

    # class axioms and decision tree axioms
    axiom_list = []
    if torch.Tensor.size(x_0.value, 0):
        p_0 = p(x_0, class_0)
        axiom_list.append(Forall(x_0, p_0))
        axiom_list.append(Forall([x_0, spache_0, cttr_0, nbxp_0], Equiv(
            Or(
                And(smaller_eq(spache_0, boundary_spache_39), smaller_eq(cttr_0, boundary_cttr_5252)),
                And(
                    And(smaller_eq(spache_0, boundary_spache_39), is_greater_than(cttr_0, boundary_cttr_5252)),
                    smaller_eq(nbxp_0, boundary_nbxp)
                )), p_0)))

    if torch.Tensor.size(x_1.value, 0):
        p_1 = p(x_1, class_1)
        axiom_list.append(Forall(x_1, p_1))
        axiom_list.append(Forall([x_1, spache_1, cttr_1, nbxp_1], Equiv(
            Or(
                Or(
                    And(
                        And(smaller_eq(spache_1, boundary_spache_39), is_greater_than(cttr_1, boundary_cttr_5252)),
                        is_greater_than(nbxp_1, boundary_nbxp)
                    ),
                    And(is_greater_than(spache_1, boundary_spache_39), smaller_eq(spache_1, boundary_spache_43))
                ),
                And(is_greater_than(spache_1, boundary_spache_43), smaller_eq(cttr_1, boundary_cttr_5268))
            ), p_1)))

    if torch.Tensor.size(x_2.value, 0):
        p_2 = p(x_2, class_2)
        axiom_list.append(Forall(x_2, p_2))
        axiom_list.append(Forall([x_2, spache_2, cttr_2], Equiv(
            And(is_greater_than(spache_2, boundary_spache_43), is_greater_than(cttr_2, boundary_cttr_5268)),
            p_2)))

    return formula_aggregator(*axiom_list)


if __name__ == "__main__":
    # Hyperparameter
    EPOCHS = 16
    BATCH_SIZE = 8
    LR = 2e-5

    # Load model, tokenizer and data
    bert = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    features = ['text', 'level', 'CTTR', 'nbXP', 'Spache']
    train = pd.read_csv("legacy/feature_first_train.csv")[features].rename(columns={'level': 'labels'})
    dev = pd.read_csv("legacy/feature_first_dev.csv")[features].rename(columns={'level': 'labels'})
    test = pd.read_csv("legacy/feature_first_test.csv")[features].rename(columns={'level': 'labels'})
    test_ua = pd.read_csv("legacy/feature_first_test_ua.csv")[features].rename(columns={'level': 'labels'})
    let = pd.read_csv("legacy/feature_first_test_let.csv")[features].rename(columns={'level': 'labels'})

    # preprocess data
    ds_train = datasets.Dataset.from_pandas(train.join(pd.DataFrame(preprocess(train).data))).with_format('torch')
    ds_dev = datasets.Dataset.from_pandas(dev.join(pd.DataFrame(preprocess(dev).data))).with_format('torch')
    ds_test = datasets.Dataset.from_pandas(test.join(pd.DataFrame(preprocess(test).data))).with_format('torch')
    ds_test_ut = datasets.Dataset.from_pandas(test_ua.join(pd.DataFrame(preprocess(test_ua).data))).with_format('torch')
    ds_test_let = datasets.Dataset.from_pandas(let.join(pd.DataFrame(preprocess(let).data))).with_format('torch')

    # create dataloaders
    train_loader = torch.utils.data.DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = torch.utils.data.DataLoader(ds_dev, batch_size=BATCH_SIZE)
    test_loader = torch.utils.data.DataLoader(ds_test, batch_size=BATCH_SIZE)
    test_ua_loader = torch.utils.data.DataLoader(ds_test_ut, batch_size=BATCH_SIZE)
    test_let_loader = torch.utils.data.DataLoader(ds_test_let, batch_size=BATCH_SIZE)

    # Predicates
    p = ltn.Predicate(LogitsToPredicate(bert))
    is_greater_than = ltn.Predicate(func=lambda a, b: a.T > b)
    smaller_eq = ltn.Predicate(func=lambda a, b: a.T <= b)

    # Constants
    class_0 = ltn.Constant(torch.tensor(0), trainable=False)
    class_1 = ltn.Constant(torch.tensor(1), trainable=False)
    class_2 = ltn.Constant(torch.tensor(2), trainable=False)

    boundary_cttr_5252 = ltn.Constant(torch.tensor(5.252), trainable=False)
    boundary_cttr_5268 = ltn.Constant(torch.tensor(5.268), trainable=False)
    boundary_nbxp = ltn.Constant(torch.tensor(11.38), trainable=False)
    boundary_spache_39 = ltn.Constant(torch.tensor(3.965), trainable=False)
    boundary_spache_43 = ltn.Constant(torch.tensor(4.395), trainable=False)

    # Operators
    Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
    And = ltn.Connective(ltn.fuzzy_ops.AndProd())
    Or = ltn.Connective(ltn.fuzzy_ops.OrProbSum())
    Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())
    Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier='f')
    Equiv = ltn.Connective(ltn.fuzzy_ops.Equiv(ltn.fuzzy_ops.AndProd(), ltn.fuzzy_ops.ImpliesReichenbach()))
    # Aggregator
    formula_aggregator = ltn.fuzzy_ops.SatAgg(ltn.fuzzy_ops.AggregPMeanError(p=2))

    # Train the model
    optimizer = AdamW(bert.parameters(), lr=LR)
    bert.to(device)

    tracker = defaultdict(list)
    best_f1 = 0
    best_model = []
    for epoch in range(EPOCHS):
        # training
        bert.train()
        for batch in tqdm(train_loader, desc='Model training', position=0):
            optimizer.zero_grad()

            # calc loss and sat, then update
            loss = bert(**transformers.BatchEncoding(
                {k: v for k, v in batch.items() if k in ['input_ids', 'attention_mask', 'token_type_ids', 'labels']}
            ).to(device)).loss
            tracker['bert_loss'].append(loss.item())
            sat = axioms(batch, training=True)
            tracker['sat_loss'].append(sat.item())
            loss += 1 - sat
            loss.backward()
            optimizer.step()
        # validation
        bert.eval()
        preds = []
        labs = []
        with torch.no_grad():
            for batch in tqdm(dev_loader):
                preds.extend(torch.argmax(bert(**transformers.BatchEncoding(
                    {k: v for k, v in batch.items() if k in ['input_ids', 'attention_mask', 'token_type_ids']}).to(
                    device)).logits, axis=-1).tolist())
                labs.extend(batch['labels'].tolist())
        acc = accuracy_score(labs, preds)
        mf1 = f1_score(labs, preds, average='macro')
        tracker['acc'].append(acc)
        tracker['mf1'].append(mf1)
        print("Acc: {:.4f}, MF1: {:.4f}".format(acc, mf1))
        if mf1 > best_f1:
            best_f1 = mf1
            bert.save_pretrained(os.path.join('legacy', 'best'))

    # save tracker
    np.save(os.path.join('legacy', 'tracker'), np.array(tracker, dtype='object'))

    # testing
    bert = AutoModelForSequenceClassification.from_pretrained(os.path.join('legacy', 'best'))
    bert.to(device)
    test_tracker = defaultdict(list)
    with torch.no_grad():
        # test set 1
        bert.eval()
        preds = []
        labs = []
        for batch in tqdm(test_loader):
            preds.extend(torch.argmax(bert(**transformers.BatchEncoding(
                {k: v for k, v in batch.items() if k in ['input_ids', 'attention_mask', 'token_type_ids']}).to(
                device)).logits, axis=-1).tolist())
            labs.extend(batch['labels'].tolist())
        acc = accuracy_score(labs, preds)
        mf1 = f1_score(labs, preds, average='macro')
        test_tracker['test_acc'].append(acc)
        test_tracker['test_mf1'].append(mf1)
        print("Test Acc: {:.4f}, MF1: {:.4f}".format(acc, mf1))

        # test set ua
        preds = []
        labs = []
        for batch in tqdm(test_ua_loader):
            preds.extend(torch.argmax(bert(**transformers.BatchEncoding(
                {k: v for k, v in batch.items() if k in ['input_ids', 'attention_mask', 'token_type_ids']}).to(
                device)).logits, axis=-1).tolist())
            labs.extend(batch['labels'].tolist())
        acc = accuracy_score(labs, preds)
        mf1 = f1_score(labs, preds, average='macro')
        test_tracker['test_ua_acc'].append(acc)
        test_tracker['test_ua_mf1'].append(mf1)
        print("Test UA Acc: {:.4f}, MF1: {:.4f}".format(acc, mf1))

        # test set let
        preds = []
        labs = []
        for batch in tqdm(test_let_loader):
            preds.extend(torch.argmax(bert(**transformers.BatchEncoding(
                {k: v for k, v in batch.items() if k in ['input_ids', 'attention_mask', 'token_type_ids']}).to(
                device)).logits, axis=-1).tolist())
            labs.extend(batch['labels'])
        labs = [{'A2': 0, 'B1': 1, 'B2': 2}[x] for x in labs]
        acc = accuracy_score(labs, preds)
        mf1 = f1_score(labs, preds, average='macro')
        test_tracker['test_let_acc'].append(acc)
        test_tracker['test_let_mf1'].append(mf1)
        print("Test LET Acc: {:.4f}, MF1: {:.4f}".format(acc, mf1))

    np.save(os.path.join('legacy', 'test_tracker'), np.array(test_tracker, dtype='object'))
