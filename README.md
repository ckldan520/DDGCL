# DDGCL
code for "Self-supervised representation learning on dynamic graphs"  [[paper]](https://dl.acm.org/doi/abs/10.1145/3459637.3482389)  


## Requirements
  Dependencies (with python >= 3.6):
  
    tensorflow==1.13.1  
    numpy==1.16.4  
    scikit_learn==1.0
    alive_progress==2.1.0


## Preprocessing

### Dataset
Create a folder 'dataset' to store data file.

[Wikipedia](http://snap.stanford.edu/jodie/wikipedia.csv)  
[Reddit](http://snap.stanford.edu/jodie/reddit.csv)  
[MOOC](http://snap.stanford.edu/jodie/mooc.csv)  


### Preprocess the data
We use the data processing method of the reference [TGAT](https://openreview.net/pdf?id=rJeW1yHYwH), [repo](https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs#inductive-representation-learning-on-temporal-graphs-iclr-2020).

We use the dense npy format to save the features in binary format. If edge features or nodes features are absent, it will be replaced by a vector of zeros.   
  
    python process.py --data wikipedia

## Model Training
Multi task learning on dynamic node classification

    python mtl_train.py wikipedia
   
Self-supervised learning on dynamic node classification

    python pre_train.py wikipedia pre_train
    python pre_train.py wikipedia fune_train
