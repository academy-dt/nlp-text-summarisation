import logging
from logging import config

import multiprocessing

from gensim import models

import itertools

from generate_bow import BowGenerator
from generate_tf_idf import TfIdfGenerator

class LdaTrainer():

    def __init__(self, tf_idf_corpus, dictionary, params):
        self.__tf_idf_corpus = tf_idf_corpus
        self.__dictionary = dictionary
        self.__params = params
        self.__model = self.__create_model(self.__tf_idf_corpus, self.__dictionary, self.__params)

    @staticmethod
    def __create_model(tf_idf_corpus, dictionary, params):
        '''Create an LDA model from the specified gen.
        The data is expected to be preprocessed.
        NOTE: It's not allowed to use generators as input, an exception is thrown:
              "Input corpus size changed during training (don't use generators as input)"
        '''
        ncpu = multiprocessing.cpu_count()
        return models.LdaMulticore(corpus=tf_idf_corpus, id2word=dictionary, workers=ncpu, **params)

    @property
    def params(self):
        return self.__params

    @property
    def model(self):
        return self.__model

def get_params_permutations(params_grid):
    grid_keys = params_grid.keys()
    grid_values = list(params_grid.values())
    grid_products = list(itertools.product(*grid_values))
    return [dict(zip(grid_keys, product)) for product in grid_products]

def train_cnn_dm_models(tf_idf_corpus, dictionary, params_grid):
    output = []
    for params in get_params_permutations(params_grid):
        logging.info(f'Training model: {params}')
        output.append(LdaTrainer(tf_idf_corpus, dictionary, params))
    return output
