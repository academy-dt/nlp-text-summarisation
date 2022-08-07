class BowGenerator():

    def __init__(self, underlying_generator, dictionary):
        self.__underlying_generator = underlying_generator
        self.__dictionary = dictionary

    @staticmethod
    def from_iterable(it, dictionary):
        return [dictionary.doc2bow(doc) for doc in it]

    def __iter__(self):
        return self

    def __next__(self):
        doc = next(self.__underlying_generator)
        return self.__dictionary.doc2bow(doc)