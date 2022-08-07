from lda_preprocessor import PreProcessor
from lda_model import LdaModel

from cnn_dm import CnnDailyMail

from scipy.special import rel_entr
from statistics import mean

import logging
from logging import config

class LdaEvaluator():

    def __init__(self, pp, model, zero_p=1e-7):
        self.__zero_p = zero_p

        self.__pp = pp
        self.__model = model

    # TODO: Consider using ELBO, reference: http://www.cs.cmu.edu/~mgormley/courses/10418/slides/lecture21-variational.pdf
    def __topic_p(self, non_zero_topics):
        topics = [self.__zero_p] * self.__model.num_topics
        for i, p in non_zero_topics:
            topics[i] = p
        return topics

    def __topics_kl(self, original_topics, summary_topics):
        original_p = self.__topic_p(original_topics)
        summary_p = self.__topic_p(summary_topics)
        return sum(rel_entr(original_p, summary_p))

    def distance(self, original, summary):
        original_topics = self.__model.predict_topics(self.__pp.process_doc(original))
        summary_topics = self.__model.predict_topics(self.__pp.process_doc(summary))
        return self.__topics_kl(original_topics, summary_topics)

def eval_cnn_dm_model(data_path, model_name):
    data_gen = CnnDailyMail.generator(data_path)

    pp = PreProcessor()
    lda = LdaModel.load(model_name)
    evaluator = LdaEvaluator(pp, lda)

    distances = [evaluator.distance(article, abstract)
                for article, abstract in data_gen]
    logging.info(f'Mean distance: {mean(distances)}')

if __name__ == '__main__':
    config.fileConfig('./logging.conf')

    import argparse
    parser = argparse.ArgumentParser(description='Evaluate an LDA model')
    parser.add_argument('data_path', type=str, help='the data to use for evaluation')
    parser.add_argument('model_name', type=str, help='The name of the model to load')
    args = parser.parse_args()

    eval_cnn_dm_model(args.data_path, args.model_name)