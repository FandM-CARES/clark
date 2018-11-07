from model import *
from process_data import *

def n_fold_test(data_list, n):
    """
    Performs an n fold test on the provided data

    Parameters:
    n (integer): number of folds

    Returns:
    Object: Mean accuracies over all the tests for each AV
    """

    data = ProcessedData(data_list)
    np_data = np.asarray(data.data)
    np.random.shuffle(np_data)
    splits = np.array_split(np_data, 10)

    mean_accuracies = {}
    for var in Model().variables:
        mean_accuracies[var] = 0

    for i, split in enumerate(splits):
        clark = Model()
        clark.train(np.concatenate(splits[:i]+splits[i+1:]))
        clark.test(splits[i])
        for var in clark.accuracies:
            mean_accuracies[var] += clark.accuracies[var]

    for var in mean_accuracies:
        mean_accuracies[var] /= n

    return mean_accuracies

def train_test_split(data_list, s):
    """
    Performs a single train-test split test

    Parameters:
    s (float): ratio of training to testing

    Returns:
    Object: Accuracy results for each AV
    """
    data = ProcessedData(data_list)
    training_data, testing_data = data.split_data([data.data])

    clark = Model()
    clark.train(training_data)
    clark.test(testing_data)
    
    return clark.accuracies