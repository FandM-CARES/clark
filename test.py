import csv

import numpy as np

from enums.global_enums import Models, NGrams
from enums.clark_enums import AV2EClassifiers
# from graphing_helpers import plot_ellsworth_figs
from models.av_model import AVModel
from models.av_to_emotion_model import AVtoEmotionModel
from models.clark_model import ClarkModel
from models.emotion_model import EmotionModel
from process_data import ProcessedData
from sklearn.naive_bayes import MultinomialNB


def n_fold_test(data_list, num_folds, model_type, compute_matrix=False, **kwargs):
    """
    Performs an n fold test on the provided data and outputs the data to a csv 

    Parameters:
    data_list (List[str]): list of paths to training data
    num_folds (int): number of folds
    model_type (Enum): denotes which model is being tested
    compute_matrix (bool): to compute confusion matrix or not. If selected, will default num_folds to 1
    **kwargs (dict): 

    Returns:
        None
    """

    if compute_matrix:
        num_folds = 2
        print("Automatically changing num_folds to 2 in order to compute confusion matrix")

    data = ProcessedData(data_list, model_type)
    np_data = np.asarray(data.data)
    np.random.shuffle(np_data)
    splits = np.array_split(np_data, num_folds)

    # plot_ellsworth_figs(np_data)

    if model_type == Models.CLARK:
        __test_CLARK_model(splits, kwargs, num_folds, compute_matrix)

    elif model_type == Models.EMOTION:
        __test_emotions_model(splits, kwargs, num_folds, compute_matrix)

    elif model_type == Models.APPRAISAL_VARIABLES:
        __test_av_model(kwargs, splits, num_folds, compute_matrix)

    elif model_type == Models.APPRAISAL_VARIABLES_TO_EMOTION:
        __test_av2e_model(splits, num_folds)
    
    else:
        raise ValueError("Unknown Model Selection")

def __test_av2e_model(splits, num_folds):
    models = ["decision_tree", "naive_bayes",
              "comp_naive_bayes", "random_forest"]
    mean_fscores = {m: [0.0, 0.0] for m in models}

    for i, split in enumerate(splits):
        av2e_model = AVtoEmotionModel()
        av2e_model.train(np.concatenate(splits[:i]+splits[i+1:]))
        av2e_model.test(splits[i])

        for m in models:
            mean_fscores[m][0] += av2e_model.micro_fscores[m]
            mean_fscores[m][1] += av2e_model.macro_fscores[m]

    for m in models:
        mean_fscores[m][0] /= num_folds
        mean_fscores[m][1] /= num_folds

    with open("results/AV2E_results.csv", "w") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Model", "Micro F Score", "Macro F Score"])
        for m in models:
            writer.writerow([m, mean_fscores[m][0], mean_fscores[m][1]])

    print("AV2E Model Done!")

def __test_av_model(kwargs, splits, num_folds, compute_matrix):
    mean_fscores = {}
    for var in AVModel(kwargs.get("ngram_choice", NGrams.UNIGRAM_AND_BIGRAM.value)).variables:
        mean_fscores[var] = [0, 0]

    for i, split in enumerate(splits):
        av = AVModel(kwargs.get("ngram_choice", NGrams.UNIGRAM_AND_BIGRAM.value))
        av.train(np.concatenate(splits[:i]+splits[i+1:]))
        av.test(splits[i])

        for var in av.variables:
            mean_fscores[var][0] += av.micro_fscores[var]
            mean_fscores[var][1] += av.macro_fscores[var]
        
        if compute_matrix:
            av.confusion_matrix("Appraisal Variables")

    for var in mean_fscores:
        mean_fscores[var][0] /= num_folds
        mean_fscores[var][1] /= num_folds

    with open("results/AV_results.csv", "w") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Variable", "Micro F Score", "Macro F Score"])
        for key, value in mean_fscores.items():
            writer.writerow([key, value[0], value[1]])

    print("AV Model Done!")

def __test_emotions_model(splits, kwargs, num_folds, compute_matrix):
    mean_fscores = [0, 0]

    for i, split in enumerate(splits):
        em_model = EmotionModel(kwargs.get("ngram_choice", NGrams.UNIGRAM_AND_BIGRAM.value))
        em_model.train(np.concatenate(splits[:i]+splits[i+1:]))
        em_model.test(splits[i])

        mean_fscores[0] += em_model.micro_fscores
        mean_fscores[1] += em_model.macro_fscores

        if compute_matrix:
            em_model.confusion_matrix("Emotions")

    mean_fscores[0] /= num_folds
    mean_fscores[1] /= num_folds

    with open("results/Emotion_results.csv", "w") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Micro F Score", "Macro F Score"])
        writer.writerow([mean_fscores[0], mean_fscores[1]])

    print("Emotions Model Done!")

def __test_CLARK_model(splits, kwargs, num_folds, compute_matrix):
    mean_fscores = [0, 0]

    for i, split in enumerate(splits):
        av2e_model = kwargs.get("av2e_classifier", AV2EClassifiers.RANDOM_FOREST.value)
        ngram_choice = kwargs.get("ngram_choice", NGrams.UNIGRAM_AND_BIGRAM.value)
        clark = ClarkModel(av2e_model, ngram_choice)
        clark.train(np.concatenate(splits[:i]+splits[i+1:]))
        clark.test(splits[i])

        mean_fscores[0] += clark.micro_fscores
        mean_fscores[1] += clark.macro_fscores

        if compute_matrix:
            clark.confusion_matrix("CLARK")

    mean_fscores[0] /= num_folds
    mean_fscores[1] /= num_folds

    with open("results/CLARK_results.csv", "w") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Micro F Score", "Macro F Score"])
        writer.writerow([mean_fscores[0], mean_fscores[1]])

    print("CLARK Model Done!")