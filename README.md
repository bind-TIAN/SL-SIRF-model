# SL-SIRF-model
---
## Introduction of .py files
---
*
    `data_analyze.py` file is used for data analyzation and visualization operation.

*    `grid.py` file contain two main module called `getGridMask` and `getSequenceGridMask`. The function of `getGridMask` used to compute the binary mask that represent the
  occupancy of each ped in the other's grid. The function `getSequenceGridMask` used to get the grid masks for all the frames in the sequence.

*    `helper.py` file contain many functions, and these functions include Gaussian distribution function, particle re-sampling function and so on.
*    `hyperparameter.py` file is a testing file which aims to validate the effectiveness of training process.
*    `model.py` file is a main model function that contains the design ideas of the structured modules necessary for the training and testing of the model.
*    `test.py` file is used for the proposed new model (without the particle filter part).
*    `test_T.py` file is used for the proposed new model (with the particle filter part).
*    `train.py` file is used for training process of our model.
*    `utils.py` file is used to load, process, and analyze data, and the processed data is generated for training and testing.
*    `visualize_T` file is used for data visualization process.
