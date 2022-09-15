# AniGAN---Generating-faces-in-the-style-of-japanese-animation

This is my first project using Keras and I decided to create a General Adversarial Neural Networks for generating anime faces. In the code, I have provided a script for scraping images from myanimelist.net. I cropped the faces out of images with the anime face detector posted here: https://github.com/nagadomi/lbpcascade_animeface. I will not include the created dataset, since I'm not sure whether I'm allowed to do so.

Results of the trained GAN are satisfying - after only 50 epochs, the generator model returns images of faces resembling anime characters. The GIF of generated images after every epoch is presented below:

![](https://github.com/mrcljns/AniGAN---Generating-faces-in-the-style-of-japanese-animation/blob/main/dcgan.gif)
