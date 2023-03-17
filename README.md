# Neural-Network-
This is a MATLAB code for building and training a neural network on the breast cancer dataset to classify cancer as malignant or benign.

* The code starts with loading and preprocessing the data. It replaces the string values in the dataset with numerical labels, checks for missing values, and removes them. 
* Then, it visualizes the data using a bar chart, a box plot, and a correlation plot.

* After that, feature extraction is done by removing some features due to multicollinearity. Then, the features are normalized, and the dataset is split into a training set, a cross-validation set, and a test set.

* The next step is to initialize the neural network's size and weights. It has one hidden layer with one unit, and the output layer has two units (one for each possible label). The weights are randomized and then converted into a single vector.

* The neural network is trained using backpropagation and a cost function, which is minimized using the fmincg function. The trained parameters (weights) are used to classify the test data, and the accuracy is calculated.
