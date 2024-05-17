# mnist-vae-pytorch
A VAE Model to generate handwritten MNIST Images

## Features
⚡Image Generation  
⚡Variational Auto Encoder (VAE)  
⚡Fully Connected Neural Network Layers  
⚡MNIST  
⚡PyTorch 

## Table of Contents
- [Introduction](#introduction) 
- [Objective](#objective)
- [Dataset](#dataset)
- [How To Use](#how-to-use)
- [Outputs](#outputs)

## Introduction
### Introduction to Variational Autoencoders (VAEs)

Variational Autoencoders (VAEs) are a type of generative model that combine the capabilities of deep learning and probabilistic inference to generate new data points similar to the input data. They work by encoding input data into a latent space, where similar data points are grouped together, and then decoding this latent representation back into the original data space. This process allows VAEs to generate new, realistic data points by sampling from the learned latent space distribution.

<img src="https://github.com/dineshg20897/mnist_vae_pytorch/blob/main/assets/VAE.png?raw=True" width="800"><br><br>

### Basic Mathematics Behind VAEs

At the core of a VAE is the concept of approximating a complex data distribution with a simpler one. The encoder network maps the input data $\( x \)$ to a latent space, producing parameters $\( \mu \)$ (mean) and $\( \sigma \)$ (standard deviation) of a Gaussian distribution. This distribution represents the posterior distribution $\( q(z|x) \)$ over the latent variables $\( z \)$. The decoder network then reconstructs the data from the latent variables by mapping $\( z \)$ back to the data space, yielding $\( p(x|z) \)$. The VAE is trained by optimizing the Evidence Lower Bound (ELBO), which consists of two main terms: the reconstruction loss, which ensures the decoder accurately reconstructs the input data, and the Kullback-Leibler (KL) divergence, which regularizes $\( q(z|x) \)$ to be close to the prior distribution $\( p(z) \)$, typically a standard normal distribution. The objective function can be written as:

```math
\mathcal{L} = \mathbb{E}_{q(z|x)} [\log p(x|z)] - \text{KL}(q(z|x) \| p(z))
```

where the first term represents the reconstruction loss and the second term is the KL divergence, promoting the latent space's regularization. Through this optimization, the VAE learns to generate new data samples by sampling from the prior distribution and decoding them via the decoder network.


## Objective

Our objective in this project is to develop a VAE and train it using the MNIST dataset, to enable the network to generate _fake_ MNIST digit images that closely resemble the images from the actual MNIST dataset.


## Dataset

The MNIST database (Modified National Institute of Standards and Technology database) is a large database of handwritten digits that is commonly used for training various image processing systems. The Images were normalized to fit into a 28x28 pixel bounding box and anti-aliased, which introduced grayscale level. It contains 60,000 training images and 10,000 testing images.


## How to use

1. Ensure the below-listed packages are installed.
    - `NumPy`
    - `matplotlib`
    - `torch`
    - `torchvision`
2. Download `VAE_using_PyTorch.ipynb` jupyter notebook from this repository.
3. Execute the notebook from start to finish in one go. If a GPU is available (recommended), it'll use it automatically; otherwise, it'll fall back to the CPU. 
4. Experiment with different hyperparameters – longer training would yield better results.


## Outputs

Here you can see the VAE generated Images after 100 epochs

<img src="https://github.com/dineshg20897/mnist_vae_pytorch/blob/main/assets/Output.png?raw=True" width="800">
