import logging
from logging import config

import multiprocessing

from gensim.corpora import Dictionary
from gensim import models

from sklearn.model_selection import GridSearchCV

import itertools

class LdaTrainer():

    def __init__(self, data, params):
        self.__data = data
        self.__params = params
        self.__dictionary = self.__get_dict(self.__data)
        self.__bow = self.__get_bow(self.__data, self.__dictionary)
        self.__tf_idf = self.__get_tf_idf(self.__bow)
        self.__model = self.__create_model(self.__dictionary, self.__tf_idf, self.__params)

    @staticmethod
    def __get_dict(data):
        '''Build dictionary from preprocessed tokenized data and remove extreme values:
        - Tokens that appear in less than 15 documents (absolute number)
        - Tokens that appear in more than 50% of the documents in the corpus
        After removal, keep only the first 100,000 most frequent tokens.
        '''
        dictionary = Dictionary(data)
        dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
        return dictionary

    @staticmethod
    def __get_bow(data, dictionary):
        '''Create Bag of Words (BoW) for each processed document.
        '''
        return [dictionary.doc2bow(doc) for doc in data]

    @staticmethod
    def __get_tf_idf(bow):
        '''Use TF-IDF (term frequency-inverse document frequency) to measure topic relevance per document.
        This is a statistical measure that evaluates how relevant a word is to a document in a collection of documents.
        This is done by multiplying two metrics:
        - How many times a word appears in a document
        - The inverse document frequency of the word across a set of documents
        '''
        tfidf = models.TfidfModel(bow)
        tfidf_corpus = tfidf[bow]
        return tfidf_corpus

    @staticmethod
    def __create_model(dictionary, tf_idf, params):
        '''Create an LDA model from the specified data.
        The data is expected to be preprocessed.
        '''
        ncpu = multiprocessing.cpu_count()
        return models.LdaMulticore(tf_idf, id2word=dictionary, workers=ncpu, **params)

    def compute_coherence(self):
        coherence = models.CoherenceModel(model=self.__model, texts=self.__data,
                                          dictionary=self.__dictionary, coherence='c_v')
        return coherence.get_coherence()

    @property
    def model(self):
        return self.__model

def grid_params(params):
    grid_keys = params.keys()
    grid_values = list(params.values())
    grid_products = list(itertools.product(*grid_values))
    return [dict(zip(grid_keys, product)) for product in grid_products]

def find_best_cnn_dm_model(data, params):
    best = (None, None)
    best_coherence = 0

    grid = grid_params(params)
    for config in grid:
        trainer = LdaTrainer(data, config)
        coherence = trainer.compute_coherence()
        logging.info(f'Model: {config} -> Coherence [{coherence}]')
        if coherence > best_coherence:
            best = (trainer, config)
    
    return best

def train_cnn_dm_model(data_path, model_path):
    logging.info('Creating CNN/DailyMail generator')
    from cnn_dm import CnnDailyMail
    cnn_dm_gen = CnnDailyMail.article_generator(data_path)

    logging.info('Creating pre-processor')
    from lda_preprocessor import PreProcessor
    pp = PreProcessor()
    pp_gen = pp.process(cnn_dm_gen)

    logging.info('Loading data')
    data = list(pp_gen)

    logging.info('Training model')
    params = {
        'num_topics': [10, 15, 20],
        'decay': [0.85, 1],
        'passes': [2],
        'alpha': [0.03, 0.05, 0.07],
        'eta': [0.03, 0.05],
        'random_state': [1024]
    }

    trainer, config = find_best_cnn_dm_model(data, params)
    logging.info(f'Found best modle: {config}')

    logging.info(f'Saving model [{model_path}]')
    trainer.model.save(model_path)

if __name__ == '__main__':
    config.fileConfig('./logging.conf')

    import argparse
    parser = argparse.ArgumentParser(description='Train an LDA model')
    parser.add_argument('data_path', type=str, help='the data to use for training')
    parser.add_argument('model_path', type=str, help='The path for the model')
    args = parser.parse_args()

    train_cnn_dm_model(args.data_path, args.model_path)