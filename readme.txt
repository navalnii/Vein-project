Vein-project is a hand vein identification using convolutional neural networks. 
There is the dataset from http://biometrics.put.poznan.pl/about-us/
       
In the prep.py file are images proccessing methods:
	Connected Components
	Skeletonize with ability choosing different kernel.
	Adaptive threshold Gaussian.

In the VGG16_based is created VGG16 based on "imagenet" model. Where the accuracy 0.95 on validation subsamples.
