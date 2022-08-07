from pointer_generator.data import example_generator

class CnnDailyMailGenerator():

    def __init__(self, data_path, single_pass=True):
        self.__underlying_generator = example_generator(data_path, single_pass)

    @staticmethod
    def __get_example_feature(example, feature):
        return example.features.feature[feature].bytes_list.value[0].decode() 

    @staticmethod
    def __get_example_article(example):
        return CnnDailyMailGenerator.__get_example_feature(example, 'article')

    @staticmethod
    def __get_example_abstract(example):
        return CnnDailyMailGenerator.__get_example_feature(example, 'abstract')

    def __iter__(self):
        return self

    def __next__(self):
        sample = next(self.__underlying_generator)
        return self.__get_example_article(sample)