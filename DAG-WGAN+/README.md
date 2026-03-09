# Efficient Generative Adversarial DAG Learning with No-Curl

Authors: Hristo Petkov and Feng Dong

Paper: Efficient Generative Adversarial DAG Learning with No-Curl

This model combines efficient structure learning and generative modelling architectures to perform faster and more accurate causal discovery.

#### Why use DAG-WGAN+?

* The combination of hybrid generative modeling for causal structure recovery with disentangled representation learning mitigates the limitations of MLE-based loss terms, resulting in higher-quality DAG-discovery.
* Refactoring the original DAG-NoCurl approach enables further refinement of the causal structure obtained from the initial estimation in search of DAG that better fit the input data. Applying this improved version of DAG-NoCurl to generative adversarial DAG-learning results in more efficient and accurate causal discovery.

#### Target Audience

The primary audience for hands-on use of DAG-WGAN+ are researchers and sophisticated practitioners in Causal Structure Learning, Probabilistic Machine Learning and AI. It is recommended to use the framework as a sequence of steps towards achieving a more accurate approximation of the generative process of data. In other words, users should focus on utilizaing the framework for their own novel research, which may include the following: 1) exploration of different Generative Models; 2) application of different Structural Causal Models; 3) integration of different data modes (e.g. time-series data, mixed data, image, video or sound data) and 4) experimentation with various architectures and hyper-parameters. We hope this framework will bridge the gap between the current state of the causal structure learning field and future contributions. 

## Introduction to DAG-WGAN+

We combine Info variational autoencoder (InfoVAE) with a generative adversarial network (GAN) to learn data probability distribution from the training data. This addresses the problems in the ELBO loss in the standard VAEs. From another perspective, we view causality learning as a process of constructing a DAG that generates data distribution equivalent to the training data. Our results demonstrate that the generative adversarial DAG training works well jointly with the InfoVAE loss to encourage mutual information between data and latent variables, leading to improved representation quality and causality discovery accuracy.

We realize the generative adversarial DAG learning based on an updated DAG-NoCurl approach to achieve both efficiency and accuracy. Our method relaxes the DAG structure obtained from the initial estimation to further
search more potential causal structures for updates. This further improves the accuracy of causal structure learning while achieving a good speed performance.

## Installation

The easiest way to gain access to our work is to clone the github repo using the following:
