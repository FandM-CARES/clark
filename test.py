from model import *
from process_data import *
import numpy as np


def n_fold_test(data_list, n, c_e):
    """
    Performs an n fold test on the provided data

    Parameters:
    data_list (ListOf str): list of paths to training data
    n (int): number of folds
    c_e (int): denotes whether using CLARK Model or Emotion Model - 0 is CLARK, 1 is Emotion

    Returns:
    Object: Mean fscores over all the tests for each AV
    """

    data = ProcessedData(data_list)
    np_data = np.asarray(data.data)
    np.random.shuffle(np_data)
    splits = np.array_split(np_data, n)

    if c_e == 0:
        mean_fscores = {}
        for var in ClarkModel().variables:
            mean_fscores[var] = [0, 0]

        for i, split in enumerate(splits):
            clark = ClarkModel()
            clark.train(np.concatenate(splits[:i]+splits[i+1:]))
            clark.test(splits[i])
            for var in clark.variables:
                mean_fscores[var][0] += clark.micro_fscores[var]
                mean_fscores[var][1] += clark.macro_fscores[var]

        for var in mean_fscores:
            mean_fscores[var][0] /= n
            mean_fscores[var][1] /= n

        with open('results/CLARK_results.csv', 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['Variable', 'Micro F Score', 'Macro F Score'])
            for key, value in mean_fscores.items():
                writer.writerow([key, value[0], value[1]])

        print('CLARK Model Done!')

    else:
        
        mean_fscores = [0, 0]

        for i, split in enumerate(splits):
            em_model = EmotionModel()
            em_model.train(np.concatenate(splits[:i]+splits[i+1:]))
            em_model.test(splits[i])

            mean_fscores[0] += em_model.micro_fscores
            mean_fscores[1] += em_model.macro_fscores

        mean_fscores[0] /= n
        mean_fscores[1] /= n

        with open('results/Emotion_results.csv', 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['Micro F Score', 'Macro F Score'])
            writer.writerow([mean_fscores[0], mean_fscores[1]])
        
        print('Emotions Model Done!')


def train_test_split(data_list, s, c_e, show_matrix=False):
    """
    Performs a single train-test split test

    Parameters:
    s (float): ratio of training to testing

    Returns:
    Object: FScore results for each AV
    """
    data = ProcessedData(data_list)
    np_data = np.asarray(data.data)
    np.random.shuffle(np_data)
    training_data, testing_data = data.split_data([data.data], s)
    if c_e == 0:
        clark = ClarkModel()
        clark.train(training_data)
        clark.test(testing_data)

        if show_matrix:
            clark.confusion_matrix()

        fscores = {}
        for var in clark.variables:
            fscores[var] = [0, 0]
            fscores[var][0] = clark.micro_fscores[var]
            fscores[var][1] = clark.macro_fscores[var]

        with open('results/CLARK_results.csv', 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['Variable', 'Micro F Score', 'Macro F Score'])
            for key, value in fscores.items():
                writer.writerow([key, value[0], value[1]])

        print('CLARK Model Done!')
    
    else:
        em_model = EmotionModel()
        em_model.train(training_data)
        em_model.test(testing_data)

        if show_matrix:
            em_model.confusion_matrix()
        
        with open('results/Emotion_results.csv', 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['Micro F Score', 'Macro F Score'])
            writer.writerow([em_model.micro_fscores, em_model.macro_fscores])
        
        print('Emotions Model Done!')
