from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier

from models.base.base_emotion_model import BaseEmotionModel


class AVtoEmotionModel(BaseEmotionModel):
    """
    """

    def __init__(self):
        super().__init__()
        self.models = ["decision_tree", "naive_bayes",
                       "comp_naive_bayes", "random_forest"]
        self.decision_tree = None
        self.naive_bayes = None
        self.comp_naive_bayes = None
        self.random_forest = None
        self.micro_fscores = {m: 0.0 for m in self.models}
        self.macro_fscores = {m: 0.0 for m in self.models}

    def train(self, training_data: list) -> None:
        self.__build_models(training_data)

    def __build_models(self, data):
        appraisal_values = [
            list(d["turn3"]["appraisals"].values()) for d in data]
        emotion_values = [em["turn3"]["emotion"] for em in data]

        clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
        clf = clf.fit(appraisal_values, emotion_values)
        self.decision_tree = clf

        naive_bayes = MultinomialNB()
        naive_bayes = naive_bayes.fit(appraisal_values, emotion_values)
        self.naive_bayes = naive_bayes
        
        comp_naive_bayes = ComplementNB()
        comp_naive_bayes = comp_naive_bayes.fit(
            appraisal_values, emotion_values)
        self.comp_naive_bayes = comp_naive_bayes

        random_forest = RandomForestClassifier(n_estimators=10)
        random_forest = random_forest.fit(appraisal_values, emotion_values)
        self.random_forest = random_forest

    def test(self, testing_data):
        appraisal_values = [list(d["turn3"]["appraisals"].values())
                            for d in testing_data]
        emotion_values = [em["turn3"]["emotion"] for em in testing_data]

        for model in self.models:
            if model == "decision_tree":
                appraisal_pred = self.decision_tree.predict(appraisal_values)
            elif model == "naive_bayes":
                appraisal_pred = self.naive_bayes.predict(appraisal_values)
            elif model == "comp_naive_bayes":
                appraisal_pred = self.comp_naive_bayes.predict(appraisal_values)
            elif model == "random_forest":
                appraisal_pred = self.random_forest.predict(appraisal_values)

            emotion_pred = classification_report(
                emotion_values, appraisal_pred, labels=self.emotions, output_dict=True)

            if "accuracy" in emotion_pred.keys():
                self.micro_fscores[model] =  emotion_pred.get("accuracy")
            else:
                self.micro_fscores[model] = emotion_pred.get("micro avg").get("f1-score")
            self.macro_fscores[model] = emotion_pred.get("macro avg").get("f1-score")
