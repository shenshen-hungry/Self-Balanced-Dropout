# Self-Balanced-Dropout
A modified dropout which can reduce the co-adaptation problem.

This project shows an example to apply Self-Balanced Dropout to CNN on SST-1 dataset.


# Requirements and preprocessing
Code is written in Python3 and requires Tensorflow (>=1.12.0).

The data preprocessing and hyper-parameter setting strictly follow the implementation in https://github.com/yoonkim/CNN_sentence [1], whose preprocessing code is reused in our project.

Pre-trained embedding 'GoogleNews-vectors-negative300.bin' is available at https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing.


# How to use
To process the raw data:
> python process_data.py vectors_path

This will create a pickle object called 'sst1.p' in the same folder, which contains the dataset in the right format.

To train the model:
> python cnn.py sst1.p

This will train and test the model.


# Reference
**[1] Yoon Kim**. Convolutional neural networks for sentence classification. EMNLP 2014.
