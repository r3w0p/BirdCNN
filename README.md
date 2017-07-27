# BirdCNN
This repository is structured as follows.

## Train
The `train` section uses TensorFlow to train a *Convolutional Neural Network* that can identify four different bird species by their distinct calls. These birds are:
- Barn Owl
- Crow
- Oriental Scops Owl
- Western Screech Owl

## Predict
The `predict` section uses the model produced in `train` to make predictions on new audio recordings of bird sounds passed to it. Any new recordings are expected to be preprocessed using the supplied `preprocess.sh` script beforehand.

## Dependencies
- Python 3
- Tensorflow
- ffmpeg
- Numpy
- sox

## Credit
Data used in `train` is a collection of primary and secondary data available in the repository [here](https://github.com/pow-pow/Datasets/tree/master/Bird%20Sounds). Secondary data is provided by [Macaulay Library](https://www.macaulaylibrary.org/) at the Cornell Lab of Ornithology, with all used data credited in their respective directories of the repository linked above.
