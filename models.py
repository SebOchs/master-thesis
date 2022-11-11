import pytorch_lightning as pl
from sklearn.metrics import f1_score, accuracy_score
import torch
from torch.optim import AdamW
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoModel


class BaselineBERT(pl.LightningModule):
    def __init__(self, batch_size=8, lr=2e-5):
        super(BaselineBERT, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=7)
        self.batch_size = batch_size
        self.lr = lr

    def forward(self, x):
        # expects input ids, attention mask and token type ids
        return torch.argmax(self.model(**x).logits, axis=-1)

    def training_step(self, batch, batch_idx):
        return self.model(**batch).loss

    def validation_step(self, batch, batch_idx):
        return {'prediction': torch.argmax(self.model(**batch).logits, axis=-1),
                'labels': batch['labels']}

    def validation_epoch_end(self, validation_step_outputs):
        all_preds = torch.cat([x['prediction'] for x in validation_step_outputs]).tolist()
        all_labels = torch.cat([x['labels'] for x in validation_step_outputs]).tolist()
        self.log('mf1', f1_score(all_labels, all_preds, average='macro'), prog_bar=True)
        self.log('acc', accuracy_score(all_labels, all_preds))

    def test_step(self, batch, batch_idx):
        return {'prediction': torch.argmax(self.model(**batch).logits, axis=-1),
                'labels': batch['labels']}

    def test_epoch_end(self, test_step_outputs):
        all_preds = torch.cat([x['prediction'] for x in test_step_outputs]).tolist()
        all_labels = torch.cat([x['labels'] for x in test_step_outputs]).tolist()
        mf1 = f1_score(all_labels, all_preds, average='macro')
        acc = accuracy_score(all_labels, all_preds)
        self.test_results = {'mf1': mf1, 'acc': acc, 'predictions': all_preds}
        self.log('mf1', f1_score(all_labels, all_preds, average='macro'), prog_bar=True)
        self.log('acc', accuracy_score(all_labels, all_preds))

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr)


class LinLayerExtension(pl.LightningModule):
    def __init__(self, batch_size=8, lr=2e-5, norm=False):
        super(LinLayerExtension, self).__init__()
        self.model = AutoModel.from_pretrained('bert-base-uncased', num_labels=7)
        self.classifier = nn.Linear(868, 7, bias=False)
        self.batch_size = batch_size
        self.lr = lr
        self.norm = norm

    def forward(self, x):
        rules = x.pop("rules")
        if self.norm:
            rules = nn.functional.normalize(rules)
        bert_output = self.model(**x).pooler_output
        extended = torch.cat([bert_output, rules], dim=-1)
        logits = self.classifier(extended)
        predictions = torch.argmax(logits, axis=-1)
        return predictions, logits, bert_output, rules

    def training_step(self, batch, batch_idx):
        rules = batch.pop("rules")
        if self.norm:
            rules = nn.functional.normalize(rules)
        labels = batch.pop("labels")
        bert_output = self.model(**batch).pooler_output
        extended = torch.cat([bert_output, rules], dim=-1)
        logits = self.classifier(extended)
        return F.cross_entropy(logits, labels)

    def validation_step(self, batch, batch_idx):
        rules = batch.pop("rules")
        if self.norm:
            rules = nn.functional.normalize(rules)
        labels = batch.pop("labels")
        bert_output = self.model(**batch).pooler_output
        extended = torch.cat([bert_output, rules], dim=-1)
        logits = self.classifier(extended)
        return {'prediction': torch.argmax(logits, axis=-1),
                'labels': labels}

    def validation_epoch_end(self, validation_step_outputs):
        all_preds = torch.cat([x['prediction'] for x in validation_step_outputs]).tolist()
        all_labels = torch.cat([x['labels'] for x in validation_step_outputs]).tolist()
        self.log('mf1', f1_score(all_labels, all_preds, average='macro'), prog_bar=True)
        self.log('acc', accuracy_score(all_labels, all_preds))

    def test_step(self, batch, batch_idx):
        rules = batch.pop("rules")
        if self.norm:
            rules = nn.functional.normalize(rules)
        labels = batch.pop("labels")
        bert_output = self.model(**batch).pooler_output
        extended = torch.cat([bert_output, rules], dim=-1)
        logits = self.classifier(extended)
        return {'prediction': torch.argmax(logits, axis=-1),
                'logits': logits,
                'bert_output': bert_output,
                'rules': rules,
                'labels': labels}

    def test_epoch_end(self, test_step_outputs):
        all_preds = torch.cat([x['prediction'] for x in test_step_outputs]).tolist()
        all_labels = torch.cat([x['labels'] for x in test_step_outputs]).tolist()
        all_logits = torch.cat([x['logits'] for x in test_step_outputs]).tolist()
        all_bert_outputs = torch.cat([x['bert_output'] for x in test_step_outputs]).tolist()
        all_rules = torch.cat([x['rules'] for x in test_step_outputs]).tolist()

        # measure rule impact
        bert_impact_pred, rule_impact_pred, bert_impact_true, rule_impact_true, rule_impact_post = [], [], [], [], []
        weights = self.classifier.weight.to('cpu')
        for pred, lab, logits, bert_output, rules in zip(all_preds, all_labels, all_logits, all_bert_outputs,
                                                         all_rules):
            max_logit = logits[pred]
            bert_impact = (torch.matmul(torch.Tensor(bert_output), weights[pred, :-100]) / max_logit).item()
            rule_impact = (torch.matmul(torch.Tensor(rules), weights[pred, -100:]) / max_logit).item()
            rule_impact_post.append((torch.Tensor(rules) * weights[pred, -100:]).tolist())

            if pred == lab:
                bert_impact_pred.append(bert_impact)
                bert_impact_true.append(bert_impact)
                rule_impact_pred.append(rule_impact)
                rule_impact_true.append(rule_impact)
            else:
                label_logit = logits[lab]
                bert_impact_label = (torch.matmul(torch.Tensor(bert_output), weights[lab, :-100]) / label_logit).item()
                rule_impact_label = (torch.matmul(torch.Tensor(rules), weights[lab, -100:]) / label_logit).item()
                # impact of bert component vs rules for true label
                bert_impact_pred.append(bert_impact)
                rule_impact_pred.append(rule_impact)
                bert_impact_true.append(bert_impact_label)
                rule_impact_true.append(rule_impact_label)

        self.test_results = {'predictions': all_preds,
                             'mf1': f1_score(all_labels, all_preds, average='macro'),
                             'acc': accuracy_score(all_labels, all_preds),
                             'bert_impact_prediction': bert_impact_pred,
                             'bert_impact_label': bert_impact_true,
                             'rule_impact_prediction': rule_impact_pred,
                             'rule_impact_label': rule_impact_pred,
                             'rule_sorting': rule_impact_post}
        self.log('mf1', f1_score(all_labels, all_preds, average='macro'))
        self.log('acc', accuracy_score(all_labels, all_preds))

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr)


class AddLayer(pl.LightningModule):
    def __init__(self, batch_size=8, lr=2e-5, norm=False):
        super(AddLayer, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=7)
        self.rule_encoder = nn.Linear(100, 7, bias=False)
        self.batch_size = batch_size
        self.lr = lr
        self.norm = norm

    def forward(self, x):
        rules = x.pop("rules")
        if self.norm:
            rules = nn.functional.normalize(rules)
        bert_logits = self.model(**x).logits
        rule_logits = self.rule_encoder(rules)
        predictions = torch.argmax(rule_logits + bert_logits, axis=-1)
        return predictions, bert_logits, rule_logits, rules

    def training_step(self, batch, batch_idx):
        rules = batch.pop("rules")
        if self.norm:
            rules = nn.functional.normalize(rules)
        labels = batch.pop("labels")
        bert_logits = self.model(**batch).logits
        rule_logits = self.rule_encoder(rules)
        return F.cross_entropy(bert_logits + rule_logits, labels)

    def validation_step(self, batch, batch_idx):
        rules = batch.pop("rules")
        if self.norm:
            rules = nn.functional.normalize(rules)
        labels = batch.pop("labels")
        bert_logits = self.model(**batch).logits
        rule_logits = self.rule_encoder(rules)
        return {'prediction': torch.argmax(bert_logits + rule_logits, axis=-1),
                'labels': labels}

    def validation_epoch_end(self, validation_step_outputs):
        all_preds = torch.cat([x['prediction'] for x in validation_step_outputs]).tolist()
        all_labels = torch.cat([x['labels'] for x in validation_step_outputs]).tolist()
        self.log('mf1', f1_score(all_labels, all_preds, average='macro'), prog_bar=True)
        self.log('acc', accuracy_score(all_labels, all_preds))

    def test_step(self, batch, batch_idx):
        rules = batch.pop("rules")
        if self.norm:
            rules = nn.functional.normalize(rules)
        labels = batch.pop("labels")
        bert_logits = self.model(**batch).logits
        rule_logits = self.rule_encoder(rules)
        return {'prediction': torch.argmax(bert_logits + rule_logits, axis=-1),
                'labels': labels,
                'bert_logits': bert_logits,
                'rule_logits': rule_logits,
                'rules': rules}

    def test_epoch_end(self, test_step_outputs):
        all_preds = torch.cat([x['prediction'] for x in test_step_outputs]).tolist()
        all_labels = torch.cat([x['labels'] for x in test_step_outputs]).tolist()
        all_bert_logits = torch.cat([x['bert_logits'] for x in test_step_outputs]).tolist()
        all_rule_logits = torch.cat([x['rule_logits'] for x in test_step_outputs]).tolist()
        all_rules = torch.cat([x['rules'] for x in test_step_outputs]).tolist()

        # measure rule impact
        bert_impact_pred, rule_impact_pred, bert_impact_true, rule_impact_true, rule_impact_post = [], [], [], [], []
        weights = self.rule_encoder.weight.to('cpu')
        for pred, lab, bert_logits, rule_logits, rules in zip(all_preds, all_labels, all_bert_logits, all_rule_logits,
                                                              all_rules):
            max_logit = bert_logits[pred] + rule_logits[pred]
            bert_impact = bert_logits[pred] / max_logit
            rule_impact = rule_logits[pred] / max_logit
            rule_impact_post.append((torch.Tensor(rules) * weights[pred]).tolist())

            if pred == lab:
                bert_impact_pred.append(bert_impact)
                bert_impact_true.append(bert_impact)
                rule_impact_pred.append(rule_impact)
                rule_impact_true.append(rule_impact)
            else:
                label_logit = bert_logits[lab] + rule_logits[lab]
                bert_impact_label = bert_logits[lab] / label_logit
                rule_impact_label = rule_logits[lab] / label_logit
                # impact of bert component vs rules for true label
                bert_impact_pred.append(bert_impact)
                rule_impact_pred.append(rule_impact)
                bert_impact_true.append(bert_impact_label)
                rule_impact_true.append(rule_impact_label)

        self.test_results = {'predictions': all_preds,
                             'mf1': f1_score(all_labels, all_preds, average='macro'),
                             'acc': accuracy_score(all_labels, all_preds),
                             'bert_impact_prediction': bert_impact_pred,
                             'bert_impact_label': bert_impact_true,
                             'rule_impact_prediction': rule_impact_pred,
                             'rule_impact_label': rule_impact_pred,
                             'rule_sorting': rule_impact_post}
        self.log('mf1', f1_score(all_labels, all_preds, average='macro'))
        self.log('acc', accuracy_score(all_labels, all_preds))

    def configure_optimizers(self):
        return AdamW([{"params": self.rule_encoder.parameters(), "lr": 1e-4},
                      {"params": self.model.parameters()}],
                     lr=self.lr)


class KennLayer(pl.LightningModule):
    def __init__(self, batch_size=8, lr=2e-5, norm=False):
        super(KennLayer, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=7)
        self.classifier = nn.Linear(107, 7, bias=False)
        self.batch_size = batch_size
        self.lr = lr
        self.norm = norm

    def forward(self, x):
        rules = x.pop("rules")
        if self.norm:
            rules = nn.functional.normalize(rules)
        bert_output = self.model(**x).logits
        extended = torch.cat([bert_output, rules], dim=-1)
        logits = self.classifier(extended)
        predictions = torch.argmax(logits, axis=-1)
        return predictions, bert_output, logits, rules

    def training_step(self, batch, batch_idx):
        rules = batch.pop("rules")
        if self.norm:
            rules = nn.functional.normalize(rules)
        labels = batch.pop("labels")
        bert_output = self.model(**batch).logits
        extended = torch.cat([bert_output, rules], dim=-1)
        logits = self.classifier(extended)
        return F.cross_entropy(logits, labels)

    def validation_step(self, batch, batch_idx):
        rules = batch.pop("rules")
        if self.norm:
            rules = nn.functional.normalize(rules)
        labels = batch.pop("labels")
        bert_output = self.model(**batch).logits
        extended = torch.cat([bert_output, rules], dim=-1)
        logits = self.classifier(extended)
        return {'prediction': torch.argmax(logits, axis=-1),
                'labels': labels}

    def validation_epoch_end(self, validation_step_outputs):
        all_preds = torch.cat([x['prediction'] for x in validation_step_outputs]).tolist()
        all_labels = torch.cat([x['labels'] for x in validation_step_outputs]).tolist()
        self.log('mf1', f1_score(all_labels, all_preds, average='macro'), prog_bar=True)
        self.log('acc', accuracy_score(all_labels, all_preds))

    def test_step(self, batch, batch_idx):
        rules = batch.pop("rules")
        if self.norm:
            rules = nn.functional.normalize(rules)
        labels = batch.pop("labels")
        bert_output = self.model(**batch).logits
        extended = torch.cat([bert_output, rules], dim=-1)
        logits = self.classifier(extended)
        return {'prediction': torch.argmax(logits, axis=-1),
                'labels': labels,
                'bert_logits': bert_output,
                'rules': rules,
                'last_logits': logits}

    def test_epoch_end(self, test_step_outputs):
        all_preds = torch.cat([x['prediction'] for x in test_step_outputs]).tolist()
        all_labels = torch.cat([x['labels'] for x in test_step_outputs]).tolist()
        all_bert_logits = torch.cat([x['bert_logits'] for x in test_step_outputs]).tolist()
        all_logits = torch.cat([x['last_logits'] for x in test_step_outputs]).tolist()
        all_rules = torch.cat([x['rules'] for x in test_step_outputs]).tolist()

        # measure rule impact
        bert_impact_pred, rule_impact_pred, bert_impact_true, rule_impact_true, rule_impact_post = [], [], [], [], []
        weights = self.classifier.weight.to('cpu')
        for pred, lab, bert_logits, last_logits, rules in zip(all_preds, all_labels, all_bert_logits, all_logits,
                                                              all_rules):
            max_logit = last_logits[pred]
            bert_impact = (torch.matmul(weights[pred, :-100], torch.Tensor(bert_logits)) / max_logit).item()
            rule_impact = (torch.matmul(weights[pred, -100:], torch.Tensor(rules)) / max_logit).item()
            rule_impact_post.append((torch.Tensor(rules) * weights[pred, -100:]).tolist())

            if pred == lab:
                bert_impact_pred.append(bert_impact)
                bert_impact_true.append(bert_impact)
                rule_impact_pred.append(rule_impact)
                rule_impact_true.append(rule_impact)
            else:
                label_logit = last_logits[lab]
                bert_impact_label = (torch.matmul(weights[lab, :-100], torch.Tensor(bert_logits)) / label_logit).item()
                rule_impact_label = (torch.matmul(weights[lab, -100:], torch.Tensor(rules)) / label_logit).item()
                # impact of bert component vs rules for true label
                bert_impact_pred.append(bert_impact)
                rule_impact_pred.append(rule_impact)
                bert_impact_true.append(bert_impact_label)
                rule_impact_true.append(rule_impact_label)

        self.test_results = {'predictions': all_preds,
                             'mf1': f1_score(all_labels, all_preds, average='macro'),
                             'acc': accuracy_score(all_labels, all_preds),
                             'bert_impact_prediction': bert_impact_pred,
                             'bert_impact_label': bert_impact_true,
                             'rule_impact_prediction': rule_impact_pred,
                             'rule_impact_label': rule_impact_pred,
                             'rule_sorting': rule_impact_post}
        self.log('mf1', f1_score(all_labels, all_preds, average='macro'))
        self.log('acc', accuracy_score(all_labels, all_preds))

    def configure_optimizers(self):
        return AdamW([{"params": self.classifier.parameters(), "lr": 1e-4},
                      {"params": self.model.parameters()}],
                     lr=self.lr)
