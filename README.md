# SMS Spam Classifier (Neural Network)

A text classification model built with **TensorFlow and Keras** that labels SMS messages as **ham** (a normal message) or **spam** (an advertisement or company message). Completed as part of the freeCodeCamp *Machine Learning with Python* certification.

**Goal:** correctly classify the challenge's test messages
**Result:** `[your test accuracy]%`

## Dataset

The [SMS Spam Collection](http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/) dataset, already split into train and test files. Each line holds a label (`ham` or `spam`) and the message text, separated by a tab.

## Approach

**1. Preprocessing**
- Labels are converted to numbers: ham = 0, spam = 1.
- A `TextVectorization` layer turns raw text into sequences of integers (vocabulary of 10,000 words, sequences padded or cut to 50 tokens).

**2. Model** – a `Sequential` network that takes raw strings:

```
TextVectorization
Embedding(10000, 32)
GlobalAveragePooling1D
Dense(32, ReLU)
Dropout(0.5)
Dense(1, Sigmoid)
```

- Optimizer: Adam
- Loss: binary cross-entropy
- Metric: accuracy
- Early stopping restores the best weights to reduce overfitting

**3. Prediction function** – `predict_message(text)` returns a list of two items:
1. a number between 0 and 1 (closer to 0 = ham, closer to 1 = spam)
2. the label `"ham"` or `"spam"`

```python
predict_message("how are you doing today?")
# [0.02, 'ham']
```

## Results

| Metric | Value |
|---|---|
| Test accuracy | `[xx]%` |
| Challenge test | Passed |

## Tech Stack

Python 3 · TensorFlow / Keras · pandas · NumPy · Google Colaboratory

## How to Run

1. Open the notebook in Google Colab and save a copy.
2. Run the first two cells to import libraries and download the data.
3. Add the code between those cells and the final test cell, then run them in order.
4. Run the final cell to test `predict_message`.
5. Turn on link sharing ("anyone with the link") before submitting.

## Possible Improvements

- Try an LSTM or bidirectional GRU layer
- Use pretrained embeddings (GloVe, Universal Sentence Encoder)
- Handle class imbalance with class weights
- Tune the vocabulary size and sequence length
