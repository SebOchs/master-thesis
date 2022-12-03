import benepar
from collections import defaultdict
import graphviz
from g2p_en import G2p
import itertools
from nltk.util import ngrams
import nltk
import numpy as np
import os
import pandas as pd
from sklearn import tree
from sklearn.metrics import f1_score, accuracy_score
import spacy
from spacy_syllables import SpacySyllables
import textstat
from tqdm import tqdm


def pos_lists(csv):
    """
    Creates a dictionary with POS as keys and words belonging to that POS as values from the EVP csv file
    :param csv: pd Dataframe / the EVP list
    :return: defaultdict / key: POS as strings, value: list of strings
    """
    pos_dict = defaultdict(list)
    translation = {
        'noun': 'NOUN',
        'adjective': 'ADJ',
        'adverb': 'ADV',
        'pronoun': 'PRON',
        'verb': 'VERB',
        'determiner': 'DET'
    }
    for idx, row in csv.iterrows():
        try:
            pos_dict[translation[row['Part of Speech']]].append(row['Base Word'])
        except KeyError:
            continue
    return pos_dict


def count_certain_phrase(pt_string_list, phrase):
    """
    Calculates avg. amount of a certain phrase (e.g. NP) per sentence in a parse treed text
    :param pt_string_list: list of strings / strings are syntactic parse trees
    :param phrase: string / name of phrase
    :return: float / avg. nr. of phrases in text
    """
    return np.mean([y.count(phrase) for y in pt_string_list])


def count_phrases(pt_string_list):
    """
    Calculates avg. amount of phrases in the parse trees of a text
    :param pt_string_list: list of strings / strings are syntactic parse trees
    :return: float / avg. nr. of specified phrase types in the parse trees
    """
    # phrase types
    PHRASES = ['ADJP', 'ADVP', 'CONJP', 'FRAG', 'INTJ', 'LST', 'NAC', 'NP', 'NX', 'PP', 'PRN', 'PRT', 'QP', 'RRC',
               'UCP', 'VP', 'WHADJP', 'WHAVP', 'WHNP', 'WHPP', 'X']
    return np.mean([sum([y.count(x) for x in PHRASES]) for y in pt_string_list])


def tf(text, vocab):
    """
    Calculates the frequency of words/lemmas in a text that match with the POS and at least one word in the vocab list
    :param text: spacy doc / tokenized text
    :param vocab: dictionary (k: string, v: list of strings) / result from function pos_lists
    :return: float
    """
    return sum([x.lemma_ in vocab[x.pos_] or x.text in vocab[x.pos_] for x in text]) / len(text)


def feature_calculation(path):
    """
    Calculates and adds features to the given dataframe
    :param data: pd dataframe / data set split
    :return: pd dataframe / dataframe with added features
    """

    data = pd.read_csv(path)

    # load and return data if already preprocessed
    if os.path.exists(os.path.join('legacy', 'feature_' + os.path.split(path)[-1])):
        return pd.read_csv(os.path.join('legacy', 'feature_' + os.path.split(path)[-1]))

    # transfer BNE levels to CEFR
    if 'test_let' not in path:
        translation = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 3}
        data['level'] = list(map(lambda x: translation[x], data['level']))
        data = data[data['level'] < 3]

    # load spacy model, discourse connectors, phoneme model, and add syllables and parse trees to pipeline
    nlp = spacy.load('en_core_web_lg')
    discourse_connectors = [x for x in pd.read_html('https://www.eapfoundation.com/vocab/academic/other/dcl/',
                                                    header=0)[3].DC.values if '...' not in str(x) and type(x) == str]
    g2p = G2p()
    benepar.download('benepar_en3')
    nlp.add_pipe("benepar", config={"model": "benepar_en3"})
    nlp.add_pipe("syllables", after="tagger")

    # preprocess text with spacy model
    preprocessed = [nlp(x) for x in tqdm(data.text, desc='Preprocessing texts with spacy')]

    # calculate frequencies of words belonging to A2, B1 or B2 vocabulary list
    data['a2_freq'] = [tf(y, a2_vocab) for y in preprocessed]
    data['b1_freq'] = [tf(y, b1_vocab) for y in preprocessed]
    data['b2_freq'] = [tf(y, b2_vocab) for y in preprocessed]

    # calculate avg. nr. of phonemes per word / verb, adj and adv length
    data['phoneme_avg'] = [np.nan_to_num(np.mean([len(g2p(x.text)) for x in y if x.is_alpha])) for y in
                           tqdm(preprocessed, desc="Calculating Phoneme Avg.")]
    data['avg_verb_length'] = [np.nan_to_num(np.mean([len(x.text) for x in y if x.pos_ == 'VERB']))
                               for y in tqdm(preprocessed, "Calculating Avg. Verb Length")]
    data['avg_adj_length'] = [np.nan_to_num(np.mean([len(x.text) for x in y if x.pos_ == 'ADJ']))
                              for y in tqdm(preprocessed, "Calculating Avg. Adj Length")]
    data['avg_adv_length'] = [np.nan_to_num(np.mean([len(x.text) for x in y if x.pos_ == 'ADV']))
                              for y in tqdm(preprocessed, "Calculating Avg. Adv Length")]

    # calculate avg. nr. of verbs belonging to present, past and future tense per verb in the text
    data['present'] = [len([x for x in y if x.tag_ == 'VBD' or x.tag_ == 'VBN']) /
                       len([[x for x in y if x.tag_.startswith('VB') or x.tag_.startswith('MD')]])
                       for y in preprocessed]
    data['past'] = [len([x for x in y if x.tag_ == 'VBP' or x.tag_ == 'VBZ']) /
                    len([[x for x in y if x.tag_.startswith('VB') or x.tag_.startswith('MD')]])
                    for y in preprocessed]
    data['future'] = [len([x for x in y if x.tag_ == 'MD']) /
                      len([[x for x in y if x.tag_.startswith('VB') or x.tag_.startswith('MD')]])
                      for y in preprocessed]

    # calculate corrected type-token ratio, avg. nr. of phrases / noun phrases in sentences / length of sentences /
    # height of parse tree
    data['CTTR'] = [len(set([x.text for x in y if x.is_alpha])) / len(y) * np.sqrt(len(y) / 2)
                    for y in tqdm(preprocessed, "Calculating CTTR")]
    data['nbXP'] = [count_phrases([x._.parse_string for x in y.sents])
                    for y in tqdm(preprocessed, "Calculating Avg. Phrases per Sentence")]
    data['nbNP'] = [count_certain_phrase([x._.parse_string for x in y.sents], 'NP')
                    for y in tqdm(preprocessed, "Calculating Avg. Noun Phrases per Sentence")]
    data['meanLenS'] = [np.nan_to_num(np.mean([len(x) for x in list(y.sents)]))
                        for y in tqdm(preprocessed, "Calculating Avg. Length of Sentence")]
    data['hghtTree'] = [np.nan_to_num(np.mean([nltk.tree.Tree.fromstring(x._.parse_string).height()
                                               for x in list(y.sents)]))
                        for y in tqdm(preprocessed, "Calculating Avg. Parse Tree Height")]

    # calculate avg. nr. of unique pos tag bigrams /trigrams / 4-grams in a text
    data['POS2G'] = [np.nan_to_num(np.mean([len(set(list(ngrams([w.pos_ for w in x], 2)))) for x in list(y.sents)]))
                     for y in tqdm(preprocessed, "Calculating Avg. of Unique POS Tag 2-Grams per Sentence")]
    data['POS3G'] = [np.nan_to_num(np.mean([len(set(list(ngrams([w.pos_ for w in x], 3)))) for x in list(y.sents)]))
                     for y in tqdm(preprocessed, "Calculating Avg. of Unique POS Tag 3-Grams per Sentence")]
    data['POS4G'] = [np.nan_to_num(np.mean([len(set(list(ngrams([w.pos_ for w in x], 4)))) for x in list(y.sents)]))
                     for y in tqdm(preprocessed, "Calculating Avg. of Unique POS Tag 4-Grams per Sentence")]

    # calculate avg. nr. of discourse connectors in a text
    data['DCR'] = [sum([y.text.count(x) for x in discourse_connectors]) / len(y)
                   for y in tqdm(preprocessed, "Calculating Discourse Connector Ratio")]

    # calculate several readability metrics
    data['Spache'] = [textstat.spache_readability(y) for y in data.text.values]
    data['Fog'] = [textstat.gunning_fog(y) for y in data.text.values]
    data['Flesch-Kincaid'] = [textstat.flesch_kincaid_grade(y) for y in data.text.values]
    data['Coleman-Liau'] = [textstat.coleman_liau_index(y) for y in data.text.values]
    data['ARI'] = [textstat.automated_readability_index(y) for y in data.text.values]
    data.to_csv(os.path.join('legacy', 'feature_first_' + os.path.split(path)[-1]))
    return data


def train_trees(train, dev):
    """
    Trains one decision tree classifier per feature combination of depth 3 and saves the top 100 performers on the dev
    set
    :param train: pd dataframe / train dataframe with features
    :param dev: pd dataframe / devc dataframe with features
    :return: tuple of objects / macro F1-score on dev set, decision tree object, list of features
    """

    # return top tree and print eval metrics on dev if training results file exists
    if os.path.exists('top_trees_3.npy'):
        top = np.load('top_trees_3.npy', allow_pickle=True)[0]
        print("Dev Acc.: ", accuracy_score(dev.level.values, top[1].predict(dev[list(top[2])])))
        print("Dev MF1: ", f1_score(dev.level.values, top[1].predict(dev[list(top[2])]), average='macro'))
        return top

    results = []

    # Trees of depth 3 can have 7 features at most
    for feature in tqdm(list(itertools.combinations(train.columns.values[4:], 7))):
        clf = tree.DecisionTreeClassifier(max_depth=3)
        X_train = train[list(feature)].values
        y_train = train.level.values
        X_dev = dev[list(feature)].values
        y_dev = dev.level.values
        clf.fit(X_train, y_train)
        results.append([f1_score(y_dev, clf.predict(X_dev), average='macro'), clf, feature])

    # sort results descending by macro F1-score
    top = sorted(results, key=lambda x: x[0], reverse=True)[:100]

    # save top 100 and create visualization file for top 1
    np.save('top_trees_3', np.array(top))
    tree.export_graphviz(top[0][1],
                         out_file='tree_3.dot',
                         filled=True,
                         class_names=['A2', 'B1', 'B2'],
                         feature_names=list(top[0][2]),
                         impurity=False,
                         label='none')

    return top[0]


def test_tree(clf, test, ua, let):
    """

    :param clf: tuple of objects / result from train_tree
    :param test: pd dataframe / test set ut with features
    :param ua: pd dataframe / test set ua with features
    :param let: pd dataframe / test set let with features
    :return: None
    """
    # predictions
    ut_results = clf[1].predict(test[list(clf[2])].values)
    ua_results = clf[1].predict(ua[list(clf[2])].values)
    x = let.level.values
    x[x == 'A2'], x[x == 'B1'], x[x == 'B2'], = 0, 1, 2
    let_results = clf[1].predict(let[list(clf[2])].values)

    # scores
    ut_acc = accuracy_score(test.level.values, ut_results)
    ut_mf1 = f1_score(test.level.values, ut_results, average='macro')
    ua_acc = accuracy_score(ua.level.values, ua_results)
    ua_mf1 = f1_score(ua.level.values, ua_results, average='macro')
    let_acc = accuracy_score(x.tolist(), let_results)
    let_mf1 = f1_score(x.tolist(), let_results, average='macro')

    np.save('tree_test_results', np.array([ut_acc, ut_mf1, ua_acc, ua_mf1, let_acc, let_mf1]))

    print("Accuracy UT: ", ut_acc)
    print("MF1 UT: ", ut_mf1)
    print("Accuracy UA: ", ua_acc)
    print("MF1 UA: ", ua_mf1)
    print("Accuracy LET: ", let_acc)
    print("MF1 LET: ", let_mf1)


if __name__ == "__main__":
    # cefr word lists
    a2_vocab = pos_lists(pd.read_csv('legacy/a2_vocab.csv'))
    b1_vocab = pos_lists(pd.read_csv('legacy/b1_vocab.csv'))
    b2_vocab = pos_lists(pd.read_csv('legacy/b2_vocab.csv'))

    # compute features
    feature_train = feature_calculation('dataset/bne/first_train.csv')
    feature_dev = feature_calculation('dataset/bne/first_dev.csv')
    feature_test = feature_calculation('dataset/bne/first_test.csv')
    feature_test_ua = feature_calculation('dataset/bne/first_test_ua.csv')
    feature_test_let = feature_calculation('dataset/bne/first_test_let.csv')

    # search best decision tree
    best_tree = train_trees(feature_train, feature_dev)

    # measure performance on test sets
    test_tree(best_tree, feature_test, feature_test_ua, feature_test_let)

