import logging

def get_cnn_dm_generator(data_path):
    logging.info('Creating CNN/DailyMail generator')
    from generate_cnn_dm import CnnDailyMailGenerator
    return CnnDailyMailGenerator(data_path)

def get_pp_generator(data_path):
    logging.info('Creating Pre-Processor generator')
    from generate_preprocessed import PreProcessGenerator
    return PreProcessGenerator(get_cnn_dm_generator(data_path))

def get_bow_generator(data_path, dictionary):
    logging.info('Creating BOW generator')
    from generate_bow import BowGenerator
    return BowGenerator(get_pp_generator(data_path), dictionary)

def get_tf_idf_generator(data_path, dictionary, tf_idf):
    logging.info('Creating TF-IDF generator')
    from generate_tf_idf import TfIdfGenerator
    return TfIdfGenerator(get_bow_generator(data_path, dictionary), tf_idf)