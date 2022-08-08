from pointer_generator.data import example_generator

class CnnDailyMailGenerator():

    def __init__(self, extracted_feature, data_path, single_pass):
        self.__underlying_generator = example_generator(data_path, single_pass)
        self.__extracted_feature = extracted_feature

    @staticmethod
    def __get_example_feature(example, feature):
        return example.features.feature[feature].bytes_list.value[0].decode() 

    def __iter__(self):
        return self

    def __next__(self):
        example = next(self.__underlying_generator)
        return self.__get_example_feature(example, self.__extracted_feature)
    
class CnnDailyMailArticleGenerator(CnnDailyMailGenerator):

    ARTICLE_FEATURE = 'article'

    def __init__(self, data_path, single_pass=True):
        super().__init__(self.ARTICLE_FEATURE, data_path, single_pass)

class CnnDailyMailAbstractGenerator(CnnDailyMailGenerator):

    ABSTRACT_FEATURE = 'abstract'

    def __init__(self, data_path, single_pass=True):
        super().__init__(self.ABSTRACT_FEATURE, data_path, single_pass)
