from generators import get_cnn_dm_abstract_generator, get_cnn_dm_article_generator
from lda_model import LdaModel

from scipy.special import rel_entr
from statistics import mean

import logging
from logging import config

class LdaEvaluator():

    def __init__(self, model, zero_p=1e-7):
        self.__zero_p = zero_p
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
        return sum(rel_entr(original_p, summary_p)) + sum(rel_entr(summary_p, original_p))

    def distance(self, original, summary):
        original_topics = self.__model.predict_topics(original)
        summary_topics = self.__model.predict_topics(summary)
        return self.__topics_kl(original_topics, summary_topics)
