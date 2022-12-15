# Style-Transfer-with-ImageTransformationNet in Tensorflow 2 <br />

Implementation of Perceptual Losses for Real-Time Style Transfe from the paper (Justin Johnson, Alexandre Alahi, Li Fei-Fei, [2016](https://arxiv.org/abs/1603.08155))

## Combine any content image with a specific style, with one quick forward pass to the trained network.
We trained a feed forward convolutional network to different style images each time, and here the results

<p align="center">
  <img src="https://user-images.githubusercontent.com/118340733/207288285-8f207638-4a51-4283-8011-75d5e171e93d.JPG" width="600" height="750" title="Content Image">
</p> <br />

## Description <br />

For training used 60000 images from COCOdataset for 2 epochs. 
Every input image pass through the Image Transformation Network. The output (generated image) is an input for the VGG19, where we extract the feature representations. As an input for VGG19 is also the style image, in which we want to train our network and the input images of the Image Transformation Network. We calculate the content and style losses, but now we update the values of the Image Transformation Network, instead of the white-image noise as in [previous work](https://github.com/ioankont/NeuralStyleTransfer).


<p align="center">
  <img src="https://user-images.githubusercontent.com/118340733/207846932-2a5b300c-047c-4d04-b96a-704f1fa1ecfd.JPG" width="600" height="750" title="Content Image">
</p> <br />
