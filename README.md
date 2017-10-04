# TensorFlow RNN CTC [![Build Status](https://travis-ci.org/ugnelis/tensorflow-rnn-ctc.svg?branch=master)](https://travis-ci.org/ugnelis/tensorflow-rnn-ctc)

Connectionist Temporal Classification (CTC) by using Recurrent Neural Network (RNN) in TensorFlow.

## Requirements

- Python 2.7+ (for Linux)
- Python 3.5+ (for Windows)
- TensorFlow 1.0+
- NumPy 1.5+
- SciPy 0.12+
- python_speech_features 0.1+

## Installation

I suggest you to use [Anaconda](https://www.anaconda.com/download/). For `TensorFlow` and `python_speech_features` use `pip`:

```bash
$ activate anaconda_env_name
(anaconda_env_name)$ pip install python_speech_features
(anaconda_env_name)$ pip install --ignore-installed --upgrade tensorflow # without GPU
(anaconda_env_name)$ pip install --ignore-installed --upgrade tensorflow-gpu # with GPU
```

## Training

Run training by using `train.py` file.

```bash
python train.py
```

## License
This project is licensed under the terms of the MIT license.
