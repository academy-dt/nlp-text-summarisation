from nltk.stem import WordNetLemmatizer, SnowballStemmer

from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS

class PreProcessGenerator():

    def __init__(self, underlying_generator):
        self.__underlying_generator = underlying_generator
        self.__lemmatizer = WordNetLemmatizer()
        self.__stemmer = SnowballStemmer('english')

    def __lemmatize_stemming(self, text):
        return self.__stemmer.stem(self.__lemmatizer.lemmatize(text, pos='v'))

    def __process_doc(self, doc):
        return [self.__lemmatize_stemming(token)
                for token in simple_preprocess(doc)
                if token not in STOPWORDS and len(token) > 3]

    def __iter__(self):
        return self

    def __next__(self):
        doc = next(self.__underlying_generator)
        return self.__process_doc(doc)