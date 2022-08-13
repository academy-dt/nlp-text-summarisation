from collections.abc import Iterable

class BowProcessor():

    def __init__(self, dictionary):
        self.__dictionary = dictionary
    
    @property
    def dictionary(self):
        return self.__dictionary

    def __process(self, doc):
        return self.__dictionary.doc2bow(doc)

    def __call__(self, input):
        if isinstance(input, Iterable):
            return [self.__process(doc) for doc in input]
        else:
            return self.__process(input)

class BowGenerator():

    def __init__(self, underlying_generator, dictionary):
        self.__underlying_generator = underlying_generator
        self.__bow = BowProcessor(dictionary)

    def __iter__(self):
        return self

    def __next__(self):
        doc = next(self.__underlying_generator)
        return self.__bow(doc)
