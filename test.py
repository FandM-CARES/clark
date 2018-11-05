from model import *
from process_data import *

data = ProcessedData(['./data/game1_big_trial.csv'])
training_data, testing_data = data.split_data([data.data])

clark = Model()
clark.train(training_data)
clark.test(testing_data)
print(clark.accuracies)