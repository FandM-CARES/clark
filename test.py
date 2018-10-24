from model import *
from processedData import *
import re

appraisalVariables = ['Pleasantness', 'Attention', 'Control',
                          'Certainty', 'Anticipated Effort', 'Responsibililty']

pd = ProcessedData('./data/game1_big_trial.csv', appraisalVariables)