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

    data = ProcessedData(data_list, 20000)
    np_data = np.asarray(data.data)
    with open('data/train_20000.txt', 'w') as outfile:
        outfile.write("id turn1 turn2 turn3 label\n")
        for i,e in enumerate(np_data):
            outfile.write(str(i)+"\t")
            vals = list(e.values())
            all_values = "\t".join(vals[1:])
            outfile.write(all_values+"\n")

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
        
        #mean_fscores = [0, 0]

        mean = {'accuracy':0., 'microPrecision':0., 'microRecall':0., 'microF1':0.}

        for i, split in enumerate(splits):
            em_model = EmotionModel()
            em_model.train(np.concatenate(splits[:i]+splits[i+1:]))
            results = em_model.test(splits[i])
            mean['accuracy'] += results[0]
            mean['microPrecision'] += results[1]
            mean['microRecall'] += results[2]
            mean['microF1'] += results[3]

        for k in mean:
            mean[k] /= 10

        print("--------------")
        print(mean)                

            

            #mean_fscores[0] += em_model.micro_fscores
            #mean_fscores[1] += em_model.macro_fscores

        #mean_fscores[0] /= n
        #mean_fscores[1] /= n

        #with open('results/Emotion_results.csv', 'w') as csv_file:
        #    writer = csv.writer(csv_file)
        #    writer.writerow(['Micro F Score', 'Macro F Score'])
        #    writer.writerow([mean_fscores[0], mean_fscores[1]])
        
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
        
        print('Emotions Model Done!')


label2emotion = {0:"others", 1:"happy", 2: "sad", 3:"angry"}

def run_full_test(training_data_list, testing_data_list):
    training_data = np.asarray(ProcessedData(training_data_list).data)
    testing_data = np.asarray(ProcessedData(testing_data_list).data)

    em_model = EmotionModel()
    em_model.train(training_data)
    em_model.test(testing_data)

    predictions = np.argmax(em_model.pred, axis=1)

    with io.open("test_clark.txt", "w", encoding="utf8") as fout:
        fout.write('\t'.join(["id", "turn1", "turn2", "turn3", "label"]) + '\n')        
        with io.open("data/testwithoutlabels.txt", encoding="utf8") as fin:
            fin.readline()
            for lineNum, line in enumerate(fin):
                fout.write('\t'.join(line.strip().split('\t')[:4]) + '\t')
                fout.write(label2emotion[predictions[lineNum]] + '\n')
