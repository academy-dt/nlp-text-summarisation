from collections.abc import Iterable

class TfIdfProcessor():

    def __init__(self, tf_idf):
        self.__tf_idf = tf_idf

    def __process(self, doc):
        return self.__tf_idf[doc]
    
    def __call__(self, input):
        if isinstance(input, Iterable):
            return [self.__process(doc) for doc in input]
        else:
            return self.__process(input)   

class TfIdfGenerator():

    def __init__(self, underlying_generator, tf_idf):
        self.__underlying_generator = underlying_generator
        self.__tf_idf = TfIdfProcessor(tf_idf)

    def __iter__(self):
        return self

    def __next__(self):
        doc = next(self.__underlying_generator)
        return self.__tf_idf(doc)