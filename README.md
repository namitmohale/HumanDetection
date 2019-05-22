# HumanDetection

This project detects Human Beings in images using Histogram of Oriented Gradients.
This project was submitted against NYU Computer Vision course requirements.

The aim was to use train a Neural Network coded from scratch without using any libraries other than numpy and train the network on the Histogram of Oriented Gradients.

The Neural Network coded is a fully connected network with input, hidden and output layers.

<b> Histogram of oriented Gradients: </b>

The essential thought behind the histogram of oriented gradients descriptor is that local object appearance and shape within an image can be described by the distribution of intensity gradients or edge directions. The image is divided into small connected regions called cells, and for the pixels within each cell, a histogram of gradient directions is compiled. The descriptor is the concatenation of these histograms. For improved accuracy, the local histograms can be contrast-normalized by calculating a measure of the intensity across a larger region of the image, called a block, and then using this value to normalize all cells within the block. This normalization results in better invariance to changes in illumination and shadowing. L2-Norm has been used in the project to achieve normalization.
Source: https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients

Accuracy :
The project managed to achieve an Average Accuracy of 94%.
