from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


class AVtoEmotionModel(object):

    def __init__(self):
        self.emotions = ['sadness', 'joy', 'fear',
                         'anger', 'challenge', 'boredom', 'frustration']
        self.models = ['DT', 'NB', 'CNB', 'RF']
        self.decision_tree = None
        self.NB = None
        self.CNB = None
        self.RF = None
        self.micro_fscores = {m: 0.0 for m in self.models}
        self.macro_fscores = {m: 0.0 for m in self.models}
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
        dt = self.__build_models(training_data)

    def __build_models(self, data):
        X = [list(d['turn3']['appraisals'].values()) for d in data]
        y = [em['turn3']['emotion'] for em in data]

        clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
        clf = clf.fit(X, y)
        self.decision_tree = clf

        nb = MultinomialNB()
        nb = nb.fit(X, y)
        self.NB = nb

        cnb = ComplementNB()
        cnb = cnb.fit(X, y)
        self.CNB = cnb

        rf = RandomForestClassifier(n_estimators=10)
        rf = rf.fit(X, y)
        self.RF = rf

    def test(self, data):
        X = [list(d['turn3']['appraisals'].values()) for d in data]
        y = [em['turn3']['emotion'] for em in data]

        # Decision Tree
        y_pred = self.decision_tree.predict(X)
        x = classification_report(
            y, y_pred, labels=self.emotions, output_dict=True)

        self.micro_fscores['DT'] = x['micro avg']['f1-score']
        self.macro_fscores['DT'] = x['macro avg']['f1-score']

        # Multinomal NB
        y_pred = self.NB.predict(X)
        x = classification_report(
            y, y_pred, labels=self.emotions, output_dict=True)

        self.micro_fscores['NB'] = x['micro avg']['f1-score']
        self.macro_fscores['NB'] = x['macro avg']['f1-score']

        # Complement NB
        y_pred = self.NB.predict(X)
        x = classification_report(
            y, y_pred, labels=self.emotions, output_dict=True)

        self.micro_fscores['CNB'] = x['micro avg']['f1-score']
        self.macro_fscores['CNB'] = x['macro avg']['f1-score']

        # Random Forest
        y_pred = self.RF.predict(X)
        x = classification_report(
            y, y_pred, labels=self.emotions, output_dict=True)

        self.micro_fscores['RF'] = x['micro avg']['f1-score']
        self.macro_fscores['RF'] = x['macro avg']['f1-score']
