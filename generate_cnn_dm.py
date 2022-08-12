from pointer_generator.data import example_generator

class CnnDailyMailGenerator():

    def __init__(self, extractor, data_path, single_pass):
        self.__underlying_generator = example_generator(data_path, single_pass)
        self.__extractor = extractor

    @staticmethod
    def __get_example_feature(example, feature):
        return example.features.feature[feature].bytes_list.value[0].decode() 

    @staticmethod
    def _get_example_article(example):
        return CnnDailyMailGenerator.__get_example_feature(example, 'article')

    @staticmethod
    def _get_example_abstract(example):
        return CnnDailyMailGenerator.__get_example_feature(example, 'abstract')

    @staticmethod
    def _get_example_both(example):
        article = CnnDailyMailGenerator._get_example_article(example)
        abstract = CnnDailyMailGenerator._get_example_abstract(example)
        return article, abstract

    def __iter__(self):
        return self

    def __next__(self):
        example = next(self.__underlying_generator)
        return self.__extractor(example)
    
class CnnDailyMailArticleGenerator(CnnDailyMailGenerator):

    def __init__(self, data_path, single_pass=True):
        super().__init__(self._get_example_article, data_path, single_pass)

class CnnDailyMailAbstractGenerator(CnnDailyMailGenerator):

    def __init__(self, data_path, single_pass=True):
        super().__init__(self._get_example_abstract, data_path, single_pass)

class CnnDailyMailBothGenerator(CnnDailyMailGenerator):

    def __init__(self, data_path, single_pass=True):
        super().__init__(self._get_example_both, data_path, single_pass)
