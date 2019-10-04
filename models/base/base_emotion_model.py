from abc import ABC, abstractmethod

class BaseEmotionModel(ABC):
    def __init__(self):
        self.variables = ["pleasantness", "attention", "control",
                          "certainty", "anticipated_effort", "responsibility"]
        self.emotions = ["sadness", "joy", "fear",
                         "anger", "challenge", "boredom", "frustration"]
        self.true = list()
        self.pred = list()
        self.micro_scores = dict()
        self.macro_scores = dict()
    
    @abstractmethod
    def train(self, training_data: list) -> None:
        """
        Builds a trained model

        Parameters:
        training_data (array): training data used to train the model

        Returns:
        None
        """
        pass

    @abstractmethod
    def test(self, testing_data: list) -> None:
        pass