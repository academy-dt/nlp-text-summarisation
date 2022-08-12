import logging

def get_cnn_dm_abstract_generator(data_path):
    logging.info('Creating CNN/DailyMail abstract generator')
    from generate_cnn_dm import CnnDailyMailAbstractGenerator
    return CnnDailyMailAbstractGenerator(data_path)

def get_cnn_dm_article_generator(data_path):
    logging.info('Creating CNN/DailyMail article generator')
    from generate_cnn_dm import CnnDailyMailArticleGenerator
    return CnnDailyMailArticleGenerator(data_path)

def get_pp_generator(data_gen):
    logging.info('Creating pre-processor generator')
    from generate_preprocessed import PreProcessGenerator
    return PreProcessGenerator(data_gen)

def get_bow_generator(data_gen, dictionary):
    logging.info('Creating BOW generator')
    from generate_bow import BowGenerator
    return BowGenerator(get_pp_generator(data_gen), dictionary)

def get_tf_idf_generator(data_gen, dictionary, tf_idf):
    logging.info('Creating TF-IDF generator')
    from generate_tf_idf import TfIdfGenerator
    return TfIdfGenerator(get_bow_generator(data_gen, dictionary), tf_idf)