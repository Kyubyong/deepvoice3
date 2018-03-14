# Deep Voice 3

## **Work In Progress**
To check the current status, see [this](https://github.com/Kyubyong/deepvoice3/issues/9).

This is a tensorflow implementation of [DEEP VOICE 3: 2000-SPEAKER NEURAL TEXT-TO-SPEECH](https://arxiv.org/pdf/1710.07654.pdf). For now I'm focusing on single speaker synthesis.

### Data

I'm trying with [Nick Offerman's audiobook files](https://www.audible.com/pd/Fiction/The-Adventures-of-Tom-Sawyer-Audiobook/B01HQMQLWK?source_code=AUDORWS0628169HI5use) for fun and [The LJ Speech Dataset](https://keithito.com/LJ-Speech-Dataset) which in public domain.

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
  
## Papers that referenced this repo

  * [Fitting New Speakers Based on a Short Untranscribed Sample](https://arxiv.org/abs/1802.06984)

