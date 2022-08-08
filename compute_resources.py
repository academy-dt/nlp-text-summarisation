import logging
from logging import config

from gensim.corpora import Dictionary
from gensim.models import TfidfModel

from generate_bow import BowGenerator

class DictionaryComputer():

    '''
    Params:
    - Remove tokens that appear in less than 15 documents (absolute number)
    - Remove tokens that appear in more than 50% of the documents in the corpus
    - After removal, keep only the first 100,000 most frequent tokens.
    '''
    DEFAULT_PARAMS = {
        'no_below': 15,
        'no_above': 0.5,
        'keep_n': 100000
    }

    def __init__(self, corpus, filter_params=DEFAULT_PARAMS):
        self.__dictionary = self.__get_dict(corpus, filter_params)

    @staticmethod
    def __get_dict(corpus, filter_params):
        dictionary = Dictionary(corpus)
        dictionary.filter_extremes(**filter_params)
        dictionary.compactify()
        return dictionary

    @property
    def dictionary(self):
        return self.__dictionary

class TfIdfComputer():

    def __init__(self, corpus, dictionary):
        self.__dictionary = dictionary
        self.__corpus = corpus
        self.__tf_idf = self.__get_tf_idf(self.__corpus, self.__dictionary)

    @staticmethod
    def __get_tf_idf(corpus, dictionary):
        bow = BowGenerator.from_iterable(corpus, dictionary)
        return TfidfModel(corpus=bow, id2word=dictionary.id2token)

    @property
    def tf_idf(self):
        return self.__tf_idf

def compute_resources(data_path, dict_path, tf_idf_path):
    from generators import get_cnn_dm_article_generator, get_pp_generator
    cnn_dm_gen = get_cnn_dm_article_generator(data_path)
    gen = get_pp_generator(cnn_dm_gen)

    logging.info('Reading pre-processed corpus')
    corpus = list(gen)

    logging.info('Computing dictionary')
    dictionary_computer = DictionaryComputer(corpus)
    dictionary = dictionary_computer.dictionary
    logging.info(f'Dictionary size = {len(dictionary)}')

    logging.info(f'Saving dictionary [{dict_path}]')
    dictionary.save(dict_path)

    logging.info('Computing TF-IDF')
    tf_idf_computer = TfIdfComputer(corpus, dictionary)
    tf_idf = tf_idf_computer.tf_idf

    logging.info(f'Saving TF-IDF [{tf_idf_path}]')
    tf_idf.save(tf_idf_path)

if __name__ == '__main__':
    config.fileConfig('./logging.conf')

    import argparse
    parser = argparse.ArgumentParser(description='Compute a dictionary and a TF-IDF model and save them to files')
    parser.add_argument('data_path', type=str, help='The data to use for the computation')
    parser.add_argument('dict_path', type=str, help='The output path for the dictionary')
    parser.add_argument('tf_idf_path', type=str, help='The output path for the TF-IDF model')
    args = parser.parse_args()

    compute_resources(args.data_path, args.dict_path, args.tf_idf_path)