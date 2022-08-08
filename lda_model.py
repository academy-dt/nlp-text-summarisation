from gensim.test.utils import datapath
from gensim.models import LdaMulticore

class LdaModel():

    def __init__(self, model):
        self.__model = model

    @staticmethod
    def load(model_path):
        model_datapath = datapath(model_path)
        gensim_model = LdaMulticore.load(model_datapath)
        return LdaModel(gensim_model)

    @property
    def num_topics(self):
        return self.__model.num_topics

    @property
    def dictionary(self):
        return self.__model.id2word

    def predict_topics(self, bow_doc):
        return sorted(self.__model[bow_doc], reverse=True, key=lambda tup: tup[1])

    def print_topics(self):
        for idx, topic in self.__model.print_topics(-1):
            print('Topic: {}\tWords: {}'.format(idx, topic))

    def save(self, model_path):
        model_datapath = datapath(model_path)
        self.__model.save(model_datapath)