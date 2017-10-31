# Deep Voice 3

## **Work In Progress**

This is a tensorflow implementation of [DEEP VOICE 3: 2000-SPEAKER NEURAL TEXT-TO-SPEECH](https://arxiv.org/pdf/1710.07654.pdf). For now I'm focusing on single speaker synthesis.

### Data

I use [The LJ Speech Dataset](https://keithito.com/LJ-Speech-Dataset).

## File Description


  * hyperparams.py: hyper parameters
  * prepro.py: creates inputs and targets, i.e., mel spectrogram, magnitude, and dones.
  * data_load.py
  * utils.py: several custom operational functions.
  * modules.py: building blocks for the networks.
  * networks.py: encoder, decoder, and converter
  * train.py: train
  * synthesize.py: inference
  * test_sents.txt: some test sentences in the paper.
