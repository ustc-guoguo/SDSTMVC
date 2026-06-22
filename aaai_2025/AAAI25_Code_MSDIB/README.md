# Multi-aspect Self-guided Deep Information Bottleneck for Multi-modal Clustering




## Installation
Requires Python >= 3.8 (tested on 3.8)

To install the required packages, run:
```
pip install -r requirements.txt
```



## Running an experiment
In the `src` directory, run:
```
python -m models.train -c <config_name>
```
where `<config_name>` is the name of an experiment config from one of the files in `src/config/experiments/`. eg. python -m models.train -c iapr

## Evaluating an experiment
Run the evaluation script:
```Bash
python -m models.evaluate -c <config_name> \ # Name of the experiment config
                          -t <tag> \         # The unique 8-character ID assigned to the experiment when calling models.train
                        
```
