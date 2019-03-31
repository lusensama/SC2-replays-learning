# Pysc2-based Supervised Training

Built based on [replay observer](https://github.com/narhen/pysc2-replay).

This program intends to replicate the results described in DeepMind's paper [SC2LE](https://arxiv.org/pdf/1708.04782.pdf) section 5:
  - 65% accuracy in game winner prediction using Fully convolutional (FullyConv) neural network strucure (unsuccessful)
  - Simple LSTM structure achieving similar accuracy (58.5% accuracy achieved)

# Main Features

  - All built in Keras
  - A function to clean up replays that are of different versions, extremely short, and corrupted
  - A function to extract replay data into .npz file for faster training.
  - A function to check the balance of replay outcomes
  - A replica of FullyConv network structure by DeepMind in Keras
  - A custom LSTM network to predict game outcome 
  - A data-generator that can run in parallel to speed up the training process (slow)
  - A data-generator that can fectch .npz file.



# Installation

#
#### Dependencies
Follow the install instructions of [pysc2](https://github.com/deepmind/pysc2), [Keras](https://keras.io/#installation), [sc2reader](https://github.com/GraylinKim/sc2reader)

Python 3.3+ is required.

Install the dependencies and devDependencies before running any of the code in this repository.

```cmd
pip install numpy
pip install random
pip install glob
pip install s2clientprotocol
```

# Usage
Note: Hyper parameters and flags features are not yet added in, will be updated soon.
```
python mimic.py # starts the training
```

