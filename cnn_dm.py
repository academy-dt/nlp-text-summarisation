from pointer_generator.data import example_generator

class CnnDailyMail():
    @staticmethod
    def __get_example_feature(example, feature):
        return example.features.feature[feature].bytes_list.value[0].decode() 

    @staticmethod
    def __get_example_article(example):
        return CnnDailyMail.__get_example_feature(example, 'article')

    @staticmethod
    def __get_example_abstract(example):
        return CnnDailyMail.__get_example_feature(example, 'abstract')

    @staticmethod
    def __raw_generator(data_path, single_pass):
        return example_generator(data_path, single_pass)

    @staticmethod
    def generator(data_path, single_pass=True):
        for ex in CnnDailyMail.__raw_generator(data_path, single_pass=single_pass):
            yield CnnDailyMail.__get_example_article(ex), CnnDailyMail.__get_example_abstract(ex)

    @staticmethod
    def article_generator(data_path, single_pass=True):
        for ex in CnnDailyMail.__raw_generator(data_path, single_pass=single_pass):
            yield CnnDailyMail.__get_example_article(ex)