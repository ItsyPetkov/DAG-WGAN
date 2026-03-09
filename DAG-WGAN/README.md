# DAG-WGAN: Causal Structure Learning With Wasserstein Generative Adversarial Networks

Authors: Hristo Petkov,  Colin Hanley and Feng Dong

Paper: DAG-WGAN: Causal Structure Learning With Wasserstein Generative Adversarial Networks 8th International Conference on Artificial Intelligence and Applications (AIFU, March 2022)

DAG-WGAN is a model based on a hybrid generative modelling architecture capable of causal discovery from observational tabular data. 

#### Why use DAG-WGAN?

* The model simultaneously performs causal structure learning and data generation to synthesize realistic samples with preserved causality.
* The model is an extension of the original NOTEARS framework capable of working with a variety of data types.
* DAG-WGAN can work with observational data synthesized using instances of additive noise and post-nonlienar models.

#### Target Audience

The primary audience for hands-on use of DAGAF are researchers and sophisticated practitioners in Causal Structure Learning, Probabilistic Machine Learning and AI. It is recommended to use the framework as a sequence of steps towards achieving a more accurate approximation of the generative process of data. In other words, users should focus on utilizaing the framework for their own novel research, which may include the following: 1) exploration of different Generative Models; 2) application of different Structural Causal Models; 3) integration of different data modes (e.g. time-series data, mixed data, image, video or sound data) and 4) experimentation with various architectures and hyper-parameters. We hope this framework will bridge the gap between the current state of the causal structure learning field and future contributions. 

## Introduction to DAG-WGAN

Our proposed new DAG-WGAN model combines WGAN-GP with an auto-encoder. A critic (discriminator) is involved to measure the Wasserstein distance between the real and synthetic 
data. In essence, the model learns causal structure in a generative process that trains the model to realistically generate synthetic data. With the explicit modelling of learnable causal relations (i.e. DAGs), the model learns how to generate synthetic data by simultaneously optimizing the causal structure and the model parameters via end-to-end training. We compare the performance of DAG-WGAN with other models that do not involve the Wasserstein metric in order to identify the contribution from the Wasserstein metric in causal structure learning. 

According to our experiments, the new DAG-WGAN model performs better than other models by a margin in tabular data with wide columns. The model works well with both continuous and discrete data while being capable of producing less noisy and more realistic data samples. It can handle multiple data types, including linear, non-linear, continuous and discrete. We conclude that the involvement of the Wasserstein metric helps causal structure learning in the generative process. 

## Visual aid for understanding critical concepts

We provide users with helpful visualizations (TLDR version of our paper) of the main features of our framework, which include the following: 1) a diagram of our entire framework with different architectures included.

<div align="center"><img width="640" height="234" alt="image" src="https://github.com/user-attachments/assets/7a55f9dd-99ec-415e-86e9-80f4251ffa15" /></div>

