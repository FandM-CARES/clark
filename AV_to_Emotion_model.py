from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class AVtoEmotionModel(object):

    def __init__(self):
        self.emotions = ['sadness', 'joy', 'fear', 'anger', 'challenge', 'boredom', 'frustration']
        self.decision_tree = None
        self.micro_fscores = 0.0
        self.macro_fscores = 0.0
        self.true = []
        self.pred = []

    def train(self, training_data):
        '''
        Builds a trained CLARK model

        Parameters:
        training_data (array): training data used to train the model

        Returns:
        Model: a trained model
        '''
        dt = self.__build_decision_tree(training_data)


    def __build_decision_tree(self, data):
        X = [list(d['turn3']['appraisals'].values()) for d in data]
        y = [em['turn3']['emotion'] for em in data]
        
        clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
        clf = clf.fit(X,y)

        self.decision_tree = clf
        
    def test(self, data):
        X = [list(d['turn3']['appraisals'].values()) for d in data]
        y = [em['turn3']['emotion'] for em in data]

        y_pred = self.decision_tree.predict(X)
        
        x = classification_report(y, y_pred, labels=self.emotions, output_dict=True)

        self.micro_fscores = x['micro avg']['f1-score']
        self.macro_fscores = x['macro avg']['f1-score']

