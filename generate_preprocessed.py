from collections.abc import Iterable

from nltk.stem import WordNetLemmatizer, SnowballStemmer

from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS

class PreProcessor():

    def __init__(self):
        self.__lemmatizer = WordNetLemmatizer()
        self.__stemmer = SnowballStemmer('english')

    def __lemmatize_stemming(self, text):
        return self.__stemmer.stem(self.__lemmatizer.lemmatize(text, pos='v'))

    def __process(self, doc):
        return [self.__lemmatize_stemming(token)
                for token in simple_preprocess(doc)
                if token not in STOPWORDS and len(token) > 3]

    def __call__(self, input):
        if isinstance(input, Iterable):
            return [self.__process(doc) for doc in input]
        else:
            return self.__process(input)   

class PreProcessGenerator():

    def __init__(self, underlying_generator):
        self.__underlying_generator = underlying_generator
        self.__pp = PreProcessor()

    def __iter__(self):
        return self

    def __next__(self):
        doc = next(self.__underlying_generator)
        return self.__pp(doc)