class BaseEmotionModel(object):
    def __init__(self):
        self.emotions = ["sadness", "joy", "fear",
                         "anger", "challenge", "boredom", "frustration"]
        self.true = list()
        self.pred = list()
        self.micro_scores = dict()
        self.macro_scores = dict()
    
    @abstractmethod
    def train(self, training_data: list):
        pass

    @abstractmethod
    def test(self, testing_data: list):
        pass