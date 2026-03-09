# DAG-WGAN: Causal Structure Learning With Wasserstein Generative Adversarial Networks

Authors: Hristo Petkov,  Colin Hanley and Feng Dong

Paper: DAG-WGAN: Causal Structure Learning With Wasserstein Generative Adversarial Networks 8th International Conference on Artificial Intelligence and Applications (AIFU, March 2022)

DAG-WGAN is a model based on a hybrid generative modelling architecture capable of causal discovery from observational tabular data. 

#### Why use DAG-WGAN?

* The model simultaneously performs causal structure learning and data generation to synthesize realistic samples with preserved causality.
* The model is an extension of the original NOTEARS framework capable of working with a variety of data types.
* DAG-WGAN can work with observational data synthesized using instances of additive noise and post-nonlienar models.

#### Target Audience

The primary audience for hands-on use of DAG-WGAN are researchers and sophisticated practitioners in Causal Structure Learning, Probabilistic Machine Learning and AI. It is recommended to use the framework as a sequence of steps towards achieving a more accurate approximation of the generative process of data. In other words, users should focus on utilizaing the framework for their own novel research, which may include the following: 1) exploration of different Generative Models; 2) application of different Structural Causal Models; 3) integration of different data modes (e.g. time-series data, mixed data, image, video or sound data) and 4) experimentation with various architectures and hyper-parameters. We hope this framework will bridge the gap between the current state of the causal structure learning field and future contributions. 

## Introduction to DAG-WGAN

Our proposed new DAG-WGAN model combines WGAN-GP with an auto-encoder. A critic (discriminator) is involved to measure the Wasserstein distance between the real and synthetic 
data. In essence, the model learns causal structure in a generative process that trains the model to realistically generate synthetic data. With the explicit modelling of learnable causal relations (i.e. DAGs), the model learns how to generate synthetic data by simultaneously optimizing the causal structure and the model parameters via end-to-end training. We compare the performance of DAG-WGAN with other models that do not involve the Wasserstein metric in order to identify the contribution from the Wasserstein metric in causal structure learning. 

According to our experiments, the new DAG-WGAN model performs better than other models by a margin in tabular data with wide columns. The model works well with both continuous and discrete data while being capable of producing less noisy and more realistic data samples. It can handle multiple data types, including linear, non-linear, continuous and discrete. We conclude that the involvement of the Wasserstein metric helps causal structure learning in the generative process. 

## Visual aid for understanding critical concepts

We provide users with helpful visualizations (TLDR version of our paper) of the main features of our framework, which include the following: 1) a diagram of our entire framework with different architectures included.

<div align="center"><img width="640" height="234" alt="image" src="https://github.com/user-attachments/assets/7a55f9dd-99ec-415e-86e9-80f4251ffa15" /></div>

DAG-WGAN employs a hybrid architecture composed of two primary components: (1) a Variational AutoEncoder (VAE) and (2) a Wasserstein Generative Adversarial Network with Gradient Penalty (WGAN-GP). The VAE component follows the structure of the DAG-GNN model. Therefore, the key distinction between DAG-WGAN and DAG-GNN is the integration of the additional WGAN-GP architecture, which is implemented through the Discriminator module.

## Installation

The easiest way to gain access to our work is to clone the github repo using the following:

```bash
git clone https://github.com/ItsyPetkov/DAG-WGAN/tree/master
cd DAG-WGAN/DAG-WGAN
```

## Examples
Here are some basic examples to get you started:

To get started with DAGAF just execute the following:
```bash
python Main.py
python Main.py -h # This line will give you all of the arguments of the model
```

This will execute the default state of our framework, where all of its parameters have been set in the Main.py file

That being said, here are some interesting things you can do:

To run DAG-WGAN with the same data, instead of generating new data everytime (default state), you can set the SYNTHESIZE flag to 0
```bash
python Main.py --synthesize 0 # Model will not generate new data and will run with the last data generated
python Main.py # Model will generate new data because synthesize is set 1 by default
```

To run DAG-WGAN with data of different dimensions, change the values of DATA_SAMPLE_SIZE (number of rows, default:5000) and DATA_VARIABLE_SIZE (number of columns, default:10)
```bash
python Main.py --data_sample_size 2500 --data_variable_size 50
python Main.py --data_sample_size 4000 --data_variable_size 20
```

To run DAG-WGAN with different types of continuous data, change the value of GRAPH_LINEAR_TYPE (default: non_linear_2). Below is a list of all possibilities
```bash
python Main.py --graph_linear_type linear 
python Main.py --graph_linear_type nonlinear_1
python Main.py --graph_linear_type nonlinear_2
python Main.py --graph_linear_type post_nonlinear_1
python Main.py --graph_linear_type post_nonlinear_2
```

To run DAG-WGAN with benchmark, discrete data instead of continuous, change DATA_TYPE to benchmark
```bash
 python Main.py --data_type benchmark --path ./ --save_directory ./ --load_directory ./
 # Benchmarks are provided in the data folder.
```

## Citing DAG-WGAN

If you wish to use our framework, please cite the following paper:

```
@article{Petkov2022DAGWGANCS,
  title={DAG-WGAN: Causal Structure Learning With Wasserstein Generative Adversarial Networks},
  author={Hristo Petkov and Colin Hanley and Feng Dong},
  journal={ArXiv},
  year={2022},
  volume={abs/2204.00387}
}
```

## License
DAGAF is Apache-2.0 licensed, as found in the LICENSE file.
