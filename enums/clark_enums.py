from enum import Enum

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier

class AV2EClassifiers(Enum):
    DECISION_TREE = DecisionTreeClassifier(criterion="entropy", max_depth=3)
    NAIVE_BAYES = MultinomialNB()
    COMP_NAIVE_BAYES = ComplementNB()
    RANDOM_FOREST = RandomForestClassifier(n_estimators=10)
