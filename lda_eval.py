from scipy.special import rel_entr

class LdaEvaluator():

    def __init__(self, model, zero_p=1e-7):
        self.__zero_p = zero_p
        self.__model = model

    def __topic_p(self, non_zero_topics):
        topics = [self.__zero_p] * self.__model.num_topics
        for i, p in non_zero_topics:
            topics[i] = p
        return topics

    def __topics_kl(self, original_topics, summary_topics):
        original_p = self.__topic_p(original_topics)
        summary_p = self.__topic_p(summary_topics)
        return sum(rel_entr(original_p, summary_p)) + sum(rel_entr(summary_p, original_p))

    def distance(self, original, summary, minimum_p=0):
        original_topics = self.__model.predict_topics(original, minimum_p)
        summary_topics = self.__model.predict_topics(summary, minimum_p)
        return self.__topics_kl(original_topics, summary_topics)

    def __call__(self, original, summary, minimum_p=0):
        original_topics = self.__model.predict_topics(original, minimum_p)
        summary_topics = self.__model.predict_topics(summary, minimum_p)
        divergence = self.__topics_kl(original_topics, summary_topics)
        return {
            'original_topics': original_topics,
            'summary_topics': summary_topics,
            'divergence': divergence
        }
