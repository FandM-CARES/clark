import math

from models.base.base_emotion_model import BaseEmotionModel
from nlp_helpers import (determine_pronoun, determine_tense, flatten,
                         ngrams_and_remove_stop_words, normalize,
                         parts_of_speech, tokenize)


class EmotionModel(BaseEmotionModel):
    """
    TODO insert docstring here
    """

    def __init__(self, ngram_choice):
        super().__init__()
        self.ngrams = {}
        self.priors = {}
        self.micro_fscores = 0.0
        self.macro_fscores = 0.0
        self.tense = self.__init_tense()
        self.pronouns = self.__init_pronouns()
        self.ngram_choice = ngram_choice

    def __init_tense(self):
        return {
            "past": {emotion: 1 for emotion in self.emotions},
            "present": {emotion: 1 for emotion in self.emotions},
            "future": {emotion: 1 for emotion in self.emotions}
        }

    def __init_pronouns(self):
        return {
            "first": {emotion: 1 for emotion in self.emotions},
            "second": {emotion: 1 for emotion in self.emotions},
            "third": {emotion: 1 for emotion in self.emotions}
        }

    def train(self, training_data):
        """
        Builds a trained Emotions model

        Parameters:
        training_data (array): training data used to train the model

        Returns:
        Model: a trained model
        """
        words = {}
        words_vocab = set()
        tense_vocab = set()
        pronoun_vocab = set()
        words_totals = {emotion: 0 for emotion in self.emotions}
        tense_totals = {emotion: 0 for emotion in self.emotions}
        pronoun_totals = {emotion: 0 for emotion in self.emotions}

        for row in training_data:
            for turn in ["turn1", "turn2", "turn3"]:
                true_emotion = row[turn]["emotion"]
                tokenized_res = tokenize(row[turn]["text"])

                pos = parts_of_speech(tokenized_res)
                tense_vocab, tense_totals, pronoun_vocab, pronoun_totals = self.__build_pos_counts(
                    pos, tense_vocab, true_emotion, tense_totals, pronoun_vocab, pronoun_totals)

                res = ngrams_and_remove_stop_words(
                    tokenized_res, self.ngram_choice)

                words_vocab, words, words_totals = self.__build_word_counts(
                    res, words_vocab, words, true_emotion, words_totals)

        sum_totals = sum(words_totals.values())
        self.priors = {emotion: float(
            words_totals[emotion]) / float(sum_totals) for emotion in self.emotions}

        self.__calculate_probabilities(
            words, words_totals, words_vocab, tense_totals, tense_vocab, pronoun_totals, pronoun_vocab)

    def __build_word_counts(self, res, words_vocab, words, true_emotion, words_totals):
        for word in res:
            words_vocab.add(word)
            if word in words:
                words[word][true_emotion] += 1
                words_totals[true_emotion] += 1
            else:
                words[word] = {emotion: 1 for emotion in self.emotions}
                words[word][true_emotion] += 1
                words_totals[true_emotion] += 1

        return words_vocab, words, words_totals

    def __build_pos_counts(self, pos, tense_vocab, true_emotion, tense_totals, pronoun_vocab, pronoun_totals):
        for p in pos:
            p_tense = determine_tense(p)
            p_pronoun = determine_pronoun(p)
            if p_tense != "":
                tense_vocab.add(p_tense)
                self.tense[p_tense][true_emotion] += 1
                tense_totals[true_emotion] += 1
            if p_pronoun != "":
                pronoun_vocab.add(p_pronoun)
                self.pronouns[p_pronoun][true_emotion] += 1
                pronoun_totals[true_emotion] += 1

        return tense_vocab, tense_totals, pronoun_vocab, pronoun_totals

    def __calculate_probabilities(self, words, words_totals, words_vocab, tense_totals, tense_vocab, pronoun_totals, pronoun_vocab):
        """
        TODO
        """

        len_vocab = len(words_vocab)

        for word in words:
            for emotion in self.emotions:
                words[word][emotion] = float(
                    words[word][emotion])/float(words_totals[emotion]+len_vocab)

        self.ngrams = words

        for tense in self.tense:
            for emotion in self.emotions:
                self.tense[tense][emotion] = float(
                    self.tense[tense][emotion])/float(tense_totals[emotion]+len(tense_vocab))

        for pronoun in self.pronouns:
            for emotion in self.emotions:
                self.pronouns[pronoun][emotion] = float(
                    self.pronouns[pronoun][emotion])/float(pronoun_totals[emotion]+len(pronoun_vocab))

    def test(self, testing_data):
        """
        Tests the precision/recall of the model

        Parameters:
        testing_data (array): data on which to test the model

        Returns:
        Null
        """

        for row in testing_data:
            u_priors = dict(self.priors)

            tokenized_turn1 = tokenize(row["turn1"]["text"])
            tokenized_turn2 = tokenize(row["turn2"]["text"])
            tokenized_turn3 = tokenize(row["turn3"]["text"])

            conv = tokenized_turn1 + tokenized_turn2 + tokenized_turn3

            parsed_message = flatten([ngrams_and_remove_stop_words(x, self.ngram_choice) for x in [
                                     tokenized_turn1, tokenized_turn2, tokenized_turn3]])
            classification = normalize(self.__classify(
                self.ngrams, parsed_message, conv, u_priors))
            for i, e in enumerate(self.emotions):
                u_priors[e] = classification[i]

            parsed_message = ngrams_and_remove_stop_words(
                tokenized_turn1, self.ngram_choice)
            classification = normalize(self.__classify(
                self.ngrams, parsed_message, tokenized_turn1, u_priors))
            for i, e in enumerate(self.emotions):
                u_priors[e] = classification[i]

            emotion = row["turn3"]["emotion"]
            self.true.append(emotion)

            parsed_message = ngrams_and_remove_stop_words(
                tokenized_turn3, self.ngram_choice)
            classification = self.__classify(
                self.ngrams, parsed_message, tokenized_turn3, u_priors, False)

            self.pred.append(str(classification))

        self.calculate_scores()

    def __classify(self, training_dict, content, tokenized_content, priors, raw=True):
        """
        Classifies each message according to the trained model

        Parameters:
        training_dict (Object): trained model
        content (String): message to be tested
        priors (Object): priors

        Returns:
        String: classification according to the trained model
        """

        sadness = [math.log(priors["sadness"]), "sadness"]
        joy = [math.log(priors["joy"]), "joy"]
        fear = [math.log(priors["fear"]), "fear"]
        challenge = [math.log(priors["challenge"]), "challenge"]
        anger = [math.log(priors["anger"]), "anger"]
        boredom = [math.log(priors["boredom"]), "boredom"]
        frustration = [math.log(priors["frustration"]), "frustration"]

        pos = parts_of_speech(tokenized_content)
        for p in pos:
            tense = determine_tense(p)
            pronoun = determine_pronoun(p)

            if tense in self.tense:
                sadness[0] += float(math.log(self.tense[tense]["sadness"]))
                joy[0] += float(math.log(self.tense[tense]["joy"]))
                fear[0] += float(math.log(self.tense[tense]["fear"]))
                challenge[0] += float(math.log(self.tense[tense]["challenge"]))
                anger[0] += float(math.log(self.tense[tense]["anger"]))
                boredom[0] += float(math.log(self.tense[tense]["boredom"]))
                frustration[0] += float(math.log(self.tense[tense]
                                                 ["frustration"]))

            if pronoun in self.pronouns:
                sadness[0] += float(math.log(self.pronouns[pronoun]["sadness"]))
                joy[0] += float(math.log(self.pronouns[pronoun]["joy"]))
                fear[0] += float(math.log(self.pronouns[pronoun]["fear"]))
                challenge[0] += float(math.log(self.pronouns[pronoun]
                                               ["challenge"]))
                anger[0] += float(math.log(self.pronouns[pronoun]["anger"]))
                boredom[0] += float(math.log(self.pronouns[pronoun]["boredom"]))
                frustration[0] += float(math.log(self.pronouns[pronoun]
                                                 ["frustration"]))

        for word in content:
            if word in training_dict:
                sadness[0] += float(math.log(training_dict[word]["sadness"]))
                joy[0] += float(math.log(training_dict[word]["joy"]))
                fear[0] += float(math.log(training_dict[word]["fear"]))
                challenge[0] += float(math.log(training_dict[word]
                                               ["challenge"]))
                anger[0] += float(math.log(training_dict[word]["anger"]))
                boredom[0] += float(math.log(training_dict[word]["boredom"]))
                frustration[0] += float(math.log(training_dict[word]
                                                 ["frustration"]))

        if raw:
            return list(map(lambda x: x[0], [sadness, joy, fear, challenge, anger, boredom, frustration]))

        return max([sadness, joy, fear, challenge, anger, boredom, frustration], key=lambda item: item[0])[1]
