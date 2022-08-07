class TfIdfGenerator():

    def __init__(self, underlying_generator, tf_idf):
        self.__underlying_generator = underlying_generator
        self.__tf_idf = tf_idf

    @staticmethod
    def from_iterable(it, tf_idf):
        return [tf_idf[doc] for doc in it]

    def __iter__(self):
        return self

    def __next__(self):
        doc = next(self.__underlying_generator)
        return self.__tf_idf[doc]