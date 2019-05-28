# Named Entity Recognition with Tensorflow

This repo implements a NER model using Tensorflow (LSTM + CRF + chars embeddings).

__A [better implementation is available here, using `tf.data` and `tf.estimator`, and achieves an F1 of 91.21](https://github.com/guillaumegenthial/tf_ner)__

State-of-the-art performance (F1 score between 90 and 91).

Check the [blog post](https://guillaumegenthial.github.io/sequence-tagging-with-tensorflow.html)

## Task

Given a sentence, give a tag to each word. A classical application is Named Entity Recognition (NER). Here is an example

```
John   lives in New   York
B-PER  O     O  B-LOC I-LOC
```


## Model

Similar to [Lample et al.](https://arxiv.org/abs/1603.01360) and [Ma and Hovy](https://arxiv.org/pdf/1603.01354.pdf).

- concatenate final states of a bi-lstm on character embeddings to get a character-based representation of each word
- concatenate this representation to a standard word vector representation (GloVe here)
- run a bi-lstm on each sentence to extract contextual representation of each word
- decode with a linear chain CRF



## Getting started


1. Download the GloVe vectors with

```
make glove
```

Alternatively, you can download them manually [here](https://nlp.stanford.edu/projects/glove/) and update the `glove_filename` entry in `config.py`. You can also choose not to load pretrained word vectors by changing the entry `use_pretrained` to `False` in `model/config.py`.

2. Build the training data, train and evaluate the model with
```
make run
```


## Details


Here is the breakdown of the commands executed in `make run`:

1. [DO NOT MISS THIS STEP] Build vocab from the data and extract trimmed glove vectors according to the config in `model/config.py`.

```
python build_data.py
```

2. Train the model with

```
python train.py
```


3. Evaluate and interact with the model with
```
python evaluate.py
```


Data iterators and utils are in `model/data_utils.py` and the model with training/test procedures is in `model/ner_model.py`

Training time on NVidia Tesla K80 is 110 seconds per epoch on CoNLL train set using characters embeddings and CRF.



## Training Data


The training data must be in the following format (identical to the CoNLL2003 dataset).

A default test file is provided to help you getting started.


```
John B-PER
lives O
in O
New B-LOC
York I-LOC
. O

This O
is O
another O
sentence
```


Once you have produced your data files, change the parameters in `config.py` like

```
# dataset
dev_filename = "data/coNLL/eng/eng.testa.iob"
test_filename = "data/coNLL/eng/eng.testb.iob"
train_filename = "data/coNLL/eng/eng.train.iob"
```




## License

This project is licensed under the terms of the apache 2.0 license (as Tensorflow and derivatives). If used for research, citation would be appreciated.

## Performance 

### use_crf = True, use_chars = True
             precision    recall  f1-score   support

      B-MISC       0.00      0.00      0.00         4
       I-LOC       0.95      0.96      0.96      2094
      I-MISC       0.89      0.89      0.89      1264
       I-ORG       0.95      0.88      0.91      2092
       I-PER       0.97      0.98      0.97      3149
           O       0.99      1.00      1.00     42759

   micro avg       0.99      0.99      0.99     51362
   macro avg       0.79      0.78      0.79     51362
weighted avg       0.99      0.99      0.99     51362

acc 98.75 - f1 92.77

### use_crf = False, use_chars = False
Epoch 7 out of 100
703/703 [==============================] - 86s - train loss: 0.0784     
              precision    recall  f1-score   support

      B-MISC       0.00      0.00      0.00         4
       I-LOC       0.92      0.94      0.93      2094
      I-MISC       0.93      0.75      0.83      1264
       I-ORG       0.93      0.76      0.84      2092
       I-PER       0.97      0.96      0.97      3149
           O       0.98      1.00      0.99     42759

   micro avg       0.98      0.98      0.98     51362
   macro avg       0.79      0.74      0.76     51362
weighted avg       0.98      0.98      0.98     51362

f1 87.91 - acc 97.60

### use_crf = False, use_chars = False, padding_mask = False

Epoch 7 out of 100
703/703 [==============================] - 93s - train loss: 0.0823      
              precision    recall  f1-score   support

      B-MISC       0.00      0.00      0.00         4
       I-LOC       0.92      0.94      0.93      2094
      I-MISC       0.93      0.74      0.83      1264
       I-ORG       0.93      0.75      0.83      2092
       I-PER       0.97      0.96      0.97      3149
           O       0.98      1.00      0.99     42759

   micro avg       0.98      0.98      0.98     51362
   macro avg       0.79      0.73      0.76     51362
weighted avg       0.97      0.98      0.97     51362

acc 97.52 - f1 87.63
