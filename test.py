from model import *
from process_data import *
import numpy as np

def n_fold_test(data_list, n):
    """
    Performs an n fold test on the provided data

    Parameters:
    n (integer): number of folds

    Returns:
    Object: Mean fscores over all the tests for each AV
    """

    data = ProcessedData(data_list)
    np_data = np.asarray(data.data)
    np.random.shuffle(np_data)
    splits = np.array_split(np_data, n)

    mean_fscores = {}
    for var in Model().variables:
        mean_fscores[var] = [0, 0]

    for i, split in enumerate(splits):
        clark = Model()
        clark.train(np.concatenate(splits[:i]+splits[i+1:]))
        clark.test(splits[i])
        for var in clark.variables:
            mean_fscores[var][0] += clark.micro_fscores[var]
            mean_fscores[var][1] += clark.macro_fscores[var]

    for var in mean_fscores:
        mean_fscores[var][0] /= n
        mean_fscores[var][1] /= n

    with open('results/output.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Variable', 'Micro F Score', 'Macro F Score'])
        for key, value in mean_fscores.items():
            writer.writerow([key, value[0], value[1]])

    print('Done!')


def train_test_split(data_list, s, show_matrix=False):
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

    clark = Model()
    clark.train(training_data)
    clark.test(testing_data)

    if show_matrix:
        clark.confusion_matrix()

    fscores = {}
    for var in clark.variables:
        fscores[var] = [0, 0]
        fscores[var][0] = clark.micro_fscores[var]
        fscores[var][1] = clark.macro_fscores[var]

    with open('results/output.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in fscores.items():
            writer.writerow([key, value[0], value[1]])

    print('Done!')
