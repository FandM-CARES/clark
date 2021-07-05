# CLARK: Conversation-Based Lexical Affect Recognition Kit

Based on the work from SemEval-2019 paper: [link]().

### Usage
```
# To train
python -m semeval.train --input train.csv --output semeval-clark.mdl

# To predict
python -m semeval.predict --input new_work.csv --saved-model semeval-clark.mdl

# To reproduce results of paper
python -m semeval.test --input dev.csv --saved-model semeval-clark.mdl
```

#### Data input format
```
id,turn1,turn2,turn3,label
0,Hi!,How are you?,I'm good :),happy
1,Who do you think you are?, I'm just a guy, Well you suck,angry
...
```

In addition, further work has been done using appraisal variables [link](). To use this version of the CLARK, see below:

### Usage (CLARK Appraisal Variables)
```
# To train
python -m train --input train.json --output av-clark.mdl

# To predict
python -m predict --input new_work.json --saved-model av-clark.mdl

# To reproduce results of paper
python -m test --input dev.json --saved-model av-clark.mdl
```

#### Data input format
```
id,turn1,turn2,turn3,label
0,Hi!,How are you?,I'm good :),happy
1,Who do you think you are?, I'm just a guy, Well you suck,angry
...
```