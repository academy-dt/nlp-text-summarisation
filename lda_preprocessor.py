from nltk.stem import WordNetLemmatizer, SnowballStemmer

from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS

class PreProcessor():

    def __init__(self):
        self.__lemmatizer = WordNetLemmatizer()
        self.__stemmer = SnowballStemmer('english')

    def __lemmatize_stemming(self, text):
        return self.__stemmer.stem(self.__lemmatizer.lemmatize(text, pos='v'))

    def process_doc(self, doc):
        return [self.__lemmatize_stemming(token)
                for token in simple_preprocess(doc)
                if token not in STOPWORDS and len(token) > 3]

    def process(self, generator):
        for doc in generator:
            yield self.process_doc(doc)