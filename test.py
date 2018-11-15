from model import *
from process_data import *

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
        mean_fscores[var] = 0

    for i, split in enumerate(splits):
        clark = Model()
        clark.train(np.concatenate(splits[:i]+splits[i+1:]))
        clark.test(splits[i])
        for var in clark.precisions:
            mean_fscores[var] += clark.fscores[var]    

    for var in mean_fscores:
        mean_fscores[var] /= n

    return mean_fscores

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
    
    return clark.precisions