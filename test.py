from model import *
from processedData import *
import re

appraisalVariables = ['Pleasantness', 'Attention', 'Control',
                          'Certainty', 'Anticipated Effort', 'Responsibililty']

clark = Model(['./data/game1_big_trial.csv'], .8)
print(clark.accuracy)