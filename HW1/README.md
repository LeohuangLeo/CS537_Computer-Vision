HW1 is about:

Detecting keypoints in images, and
Computing deep features of the detected keypoints.
Keypoint detection
Download the following set of 10 images images.zip. In each image, detect 200 highest-response SIFT keypoints, using the OpenCV Python library:

https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_sift_intro/py_sift_intro.html
Output: The tensor of (x,y) coordinates of 200 keypoints detected in all 10 images of the given set. The tensor size is (10 x 200 x 2). Save the tensor in the file with name: keypoints.pth 

Keypoint description
For every keypoint that you detected in images from images.zip, compute a 128-dimensional deep feature using a CNN. For this, we have prepared a skeleton code for you that you will need to complete in order to achieve the task. The  skeleton code consists of two parts.

1) Patch extraction: The first skeleton code can be found at keypoint_detector.zip. It takes as input an image and (x,y) coordinates of 200 keypoints detected in the image, and then extracts 32x32 patches around each keypoint. This code is complete, and you do not need to modify it.

2)  Computing keypoint feature descriptors: The second skeleton code can be found at keypoint_descriptor.zip. An image patch (e.g., extracted around a keypoint) is input to a CNN for computing the corresponding deep feature of the patch. This code is not complete, and you will need to modify it as follows.

Design new CNN2 and CNN3, starting from the provided initial CNN1 architecture in keypoint_descriptor.zip.
Train all three networks CNN1, CNN2, and CNN3 on the provided training dataset.
Using CNN1, CNN2, and CNN3, compute the corresponding three sets of deep features of all your keypoints detected in images.zip.
The following describes each of the above tasks in more detail.

Designing the New CNNs
Design the new CNN2 and CNN3 by augmenting the CNN1 architecture provided in the skeleton code. CNN2 should have at least two new additional convolutional layers relative to CNN1, and all other layers the same as in CNN1. Also, CNN3 should have at least two new additional convolutional layers relative to CNN2. For each new convolutional layer that you add to CNN1, you would need to manually specify: size of the kernel, number of channels, stride, and other required hyper-parameters. You should empirically find optimal hyper-parameters on the training dataset so that you achieve a minimum loss in training.

Training the CNNs
The training set that you will use for training your CNNs consists of image patches,  which can be found at: http://phototour.cs.washington.edu/patches/ (Links to an external site.). You do not need to download this training set, since the skeleton code will automatically download it for you from the folder "/scratch/CS537_2020_Spring/data" on the Pelican server.

The training set organizes image patches in triplets: (anchor, positive, negative), denoted as (x, x+, x-). An anchor patch x and positive patch x+ represent the same detail of the scene but captured in two different images, and hence they should have very similar feature descriptors. An anchor patch x and negative patch x- represent two different details from two different scenes, and hence they should not have similar feature descriptors. Therefore, for training our CNNs, we can use these training triplets, and seek to estimate all CNN parameters by minimizing the following triplet loss function for every triplet :

L = max(0, d(x,x+) - d(x,x-) + m)

where d() is a distance between deep features of the corresponding patches computed by the CNN, specified as:

d(x,x') = || f(x)-f(x') ||^2

and m>0 is a margin set to m=0.1.

You do not need to implement this loss function, since it is already provided in the skeleton code. Again, the skeleton code also handles selection of patches into triplets, so you do not need to implement the data loader for forming the triplets.

Training of a CNN is conducted by iterating over many epochs, where in each epoch randomly selects mini-batches of triplets. Your task will be to select a suitable: learning rate, size of the mini-batch, and the number of epochs for training.

Output: Four plots of the training loss over more than 20 epochs, for two different values of mini-batch size and two different learning rates, where one of these four plots should have the optimal hyper-parameters that you have empirically found.

Computing Deep Features
Once you train CNN1, CNN2, and CNN3 on the training dataset, you will need to run the provided skeleton code for extracting patches around (x,y) locations of keypoints detected in images.zip, and then pass these patches to CNN1, CNN2, and CNN3 for computing the corresponding deep features. You do not need to implement the code for saving the results in a file, since the skeleton code will do this for you. The deep features will be saved in: features_CNN*.pth  (e.g., for CNN1 the file name will be features_CNN1.pth) 

Output: Three sets of deep features in features_CNN*.pth, produced by the trained CNN1, CNN2, CNN3, of all your keypoints detected in images.zip.

What To Turn In?
A compressed folder name.tar.gz (please use the Linux command tar for compression) that contains the following files:

(5  points) A one-page pdf showing the four plots of the training loss for each CNN over at least 20 epochs for two values of mini-batch size and two learning rates (a total  of 12 plots for CNN1, CNN2, and CNN3). Clearly mark the plots with the optimal hyper-parameters that you have empirically found.
(5 points) keypoints.pth,  with the tensor of (x,y) coordinates of 200 keypoints detected in the 10 images of images.zip.
(5 points) Pytorch source code with your implementation of CNN1, CNN2, and CNN3.  
(5 points) The three output files features_CNN1.pth, features_CNN2.pth, features_CNN3.pth. Importantly, the deep features in these files must be in the same order as  (x,y) coordinates in the corresponding keypoints.pth file.
Please double-check that you can uncompress your folder with the Linux command: tar  -xzf 

Submit your compressed folder on TEACH (Links to an external site.) before 8am on April 21.

Grading
We will evaluate the accuracy of your keypoint detection and description. For the former, we will compute a Euclidean distance between (x,y) coordinates of your keypoints and (x,y) coordinates of our keypoints for the 10 images of images.zip. Since we will also use the OpenCV Python library for keypoint detection, the Euclidean distance between our keypoints and yours should be zero. We will rank all students based on this Euclidean distance and give partial credit for non-zero distances according to this ranking.

For evaluating your keypoint description, for each image in images.zip, we will form pairs of keypoints detected in that image, and compute the following hinge loss using the corresponding deep features (f_i, f_j) from your files  features_CNN*.pth:

 

eq.png

where m_n=0.2 and m_p =1 are margins. We will rank all students based on this hinge loss and give partial credit for large hinge loss values according to this ranking.

Rubric
20 points = If you submit everything from the above turn-in list,
30 points = If (x,y) coordinates of your keypoints are accurately computed, in terms of the Euclidean distance from ours
50 points = If your deep features are accurately computed, in terms of the hinge loss L
