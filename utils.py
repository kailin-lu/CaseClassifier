import re
import string
import itertools
from zipfile import ZipFile

import matplotlib.pyplot as plt 
import numpy as np

from nltk import tokenize

from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Use a tokenizer that retains punctuation
tokenizer = tokenize.WordPunctTokenizer()

# Sentences data 
zfile = ZipFile('data/sentences.zip')
    
################################################################
# Load data functions 
################################################################

def get_all_cases(caseids):
    items = zfile.namelist()
    caseids = list(caseids)
    cases = [None] * len(caseids)
    for item in items:
        if 'contentMajOp' in item:
            _,_year,fname = item.split('/')
            _, year = _year.split('_')
            caseid,_,_ = fname.split('_')
            if caseid in caseids:
                idx = caseids.index(caseid)
                cases[idx] = item
    return cases

def build_corpus(cases, labels, geniss=None, topic_filter=None):
    """
    :return: List of corpus documents, list of labels
    """
    assert len(labels) == len(geniss)
    # Find NoneType in training major ops and remove from labels
    labels = [label for i, label in enumerate(labels) if cases[i] is not None]
    geniss = [topic for i, topic in enumerate(geniss) if cases[i] is not None]
    #idx = [i for i, case in enumerate(cases) if cases[i] is not None]
    cases = [item for item in cases if item is not None]
    #filtered_additional = additional.iloc(idx)
    if topic_filter:
        idx = [i for i, gen in enumerate(geniss) if geniss[i]==topic_filter]
        labels = [label for i, label in enumerate(labels) if geniss[i]==topic_filter] 
        cases = [case for i, case in enumerate(cases) if geniss[i]==topic_filter]
        #filtered_additional = filtered_additional.iloc(idx)
    print('Num Cases: {}, Num Labels: {}'.format(len(cases), len(labels)))
    caseids = [item.split('/')[2].split('_')[0] for item in cases]
    cases = [' '.join(doc) for doc in document_iterator(cases)]
    assert len(cases) == len(labels) 
    return cases, labels, caseids


def load_ngrams(corpus, ngram_range, tfidf=True, max_features=10000): 
    """
    Convert corpus into bag of words matrix 
    
    :param corpus: list of documents 
    :param ngram_range: tuple of min and max ngrams to use 
    :param tfidf: if True, return TF-IDF weighted matrix 
    :param max_features: maximum vocabulary size 
    :return: sparse matrix X, fitted vectorizer 
    """
    if tfidf: 
        vectorizer = TfidfVectorizer(ngram_range=ngram_range, 
                                     stop_words='english', 
                                     strip_accents='ascii',
                                     max_df=0.95, 
                                     min_df=10,
                                     max_features=max_features)
    else: 
        vectorizer = CountVectorizer(ngram_range=ngram_range, 
                                     stop_words='english', 
                                     strip_accents='ascii',
                                     max_df=0.95,
                                     min_df=10,
                                     max_features=max_features)
    X = vectorizer.fit_transform(corpus) 
    return X, vectorizer

################################################################
# Visualization functions 
################################################################  

def plot_confusion_matrix(val_pred, val_truth, classes=[-1, 1],
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    From sklearn documentation 
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = confusion_matrix(val_pred, val_truth)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=0)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > .5 else "black")

    plt.tight_layout()
    plt.ylabel('Predicted label')
    plt.xlabel('True label')
    
    
################################################################
# Helper functions 
################################################################

def one_hot_labels(labels): 
    """
    Convert labels array to one hot vectors 
    """
    classes = len(set(labels)) 
    Y_onehot = np.zeros((len(labels), classes))      
    Y_onehot[np.arange(len(labels)), labels] = 1 
    return Y_onehot


def trim_and_pad(sequence, max_seq_length): 
    """
    Trim off end if sequence > max_seq_length 
    Pad with 0 if sequence < max_seq_length
    """

    if len(sequence) >= max_seq_length: 
        return sequence[:max_seq_length] 
    else: 
        diff = max_seq_length - len(sequence) 
        pad = np.zeros(diff, dtype=int) 
        return np.concatenate((sequence, pad))



def document_iterator(items):
    """
    Iterate through major opinion documents 
    """
    
    # Local path of all of our document files    
    for count, item in enumerate(items):
            
        if 'contentMajOp' not in item:
            continue
        _,_year,fname = item.split('/')
        _, year = _year.split('_')
       
        regex = re.compile('[%s]|\n' % re.escape(string.punctuation))
        txt = zfile.open(item).read().decode().lower()
        txt = regex.sub(' ', txt)
        tokens = tokenizer.tokenize(txt)
        
        # Keep only tokens greater than 1 characters
        # Remove first 5 words from whole doc
        tokens = [token for token in tokens if len(token)>1
                  and token not in tokens[:5]]
        yield tokens
