# CLARK: Conversation-Based Lexical Affect Recognition Kit

Based on the work from SemEval-2019 paper: [CLARK at SemEval-2019 Task 3: Exploring the Role of Context toIdentify Emotion in a Short Conversation](https://aclanthology.org/S19-2024.pdf).

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

In addition, further work has been done using appraisal variables [Patterns of cognitive appraisal in emotion](https://pubmed.ncbi.nlm.nih.gov/3886875/). To use this version of the CLARK, see below:

### Usage (CLARK Appraisal Variables)
```
# To train
python -m train --input train.json --output av-clark.mdl

# To predict
python -m predict --input new_work.json --saved-model av-clark.mdl

# To reproduce results of paper
python -m test --input dev.json --saved-model av-clark.mdl
```

#### Conversation data input format:
```
{
        
    "id":2,
    "turn1":"Hi!",
    "turn2":"How are you",
    "turn3":"I'm good :)",
    "label":"happy",
    "numUtterances":3,
    "hitCountEmotion":5,
    "hitCountAppraisal":8
},
...
```

#### Appraisal variable data input format:
```
[
    {
        "id": 12,
        "turn1": 
        {
            "emotion": "joy", 
            "appraisals": {
                "pleasantness": 2.0, "attention": 1.0, "control": 1.0, "certainty": 2.0, "anticipated_effort": 1.0, "responsibility": 1.0
            }
        }, 
        "turn2": 
        {
            "emotion": "joy", 
            "appraisals": {
                "pleasantness": 2.0, "attention": 1.0, "control": 2.0, "certainty": 2.0, "anticipated_effort": 1.0, "responsibility": 1.0
            }
        }, 
        "turn3": 
        {
            "emotion": "challenge", 
            "appraisals": {
                "pleasantness": 1.0, "attention": 0.0, "control": 2.0, "certainty": 0.0, "anticipated_effort": 0.0, "responsibility": 2.0
            }
        }, 
        "fleiss_kappa": 0.44148936170212755
        
        }, 
    ...
]
```
