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

```bash
git clone https://github.com/ItsyPetkov/DAG-WGAN/tree/master
cd DAG-WGAN/DAG-WGAN+
```
## Examples
Here are some basic examples to get you started:

To get started with DAG-WGAN+ just execute the following:
```bash
python Main.py
python Main.py -h # This line will give you all of the arguments of the model
```

This will execute the default state of our framework, where all of its parameters have been set in the Main.py file

That being said, here are some interesting things you can do:

To run DAG-WGAN+ with the same data, instead of generating new data everytime (default state), you can set the SYNTHESIZE flag to 0
```bash
python Main.py --synthesize 0 # Model will not generate new data and will run with the last data generated
python Main.py # Model will generate new data because synthesize is set 1 by default
```

To run DAG-WGAN+ with data of different dimensions, change the values of DATA_SAMPLE_SIZE (number of rows, default:5000) and DATA_VARIABLE_SIZE (number of columns, default:10)
```bash
python Main.py --data_sample_size 2500 --data_variable_size 50
python Main.py --data_sample_size 4000 --data_variable_size 20
```

To run DAG-WGAN+ with different types of continuous data, change the value of GRAPH_LINEAR_TYPE (default: non_linear_2). Below is a list of all possibilities
```bash
python Main.py --graph_linear_type linear 
python Main.py --graph_linear_type nonlinear_1
python Main.py --graph_linear_type nonlinear_2
python Main.py --graph_linear_type post_nonlinear_1
python Main.py --graph_linear_type post_nonlinear_2
```

To run DAG-WGAN+ with benchmark, discrete data instead of continuous, change DATA_TYPE to benchmark
```bash
 python Main.py --data_type benchmark --path ./ --save_directory ./ --load_directory ./
 # Benchmarks are provided in the data folder.
```

## Citing DAG-WGAN+

If you wish to use our framework, please cite the following paper:

```
@article{Petkov2023EfficientGA,
  title={Efficient Generative Adversarial DAG Learning with No-Curl},
  author={Hristo Petkov and Feng Dong},
  journal={2023 International Conference Automatics and Informatics (ICAI)},
  year={2023},
  pages={164-169}
}
```

## License
DAG-WGAN+ is Apache-2.0 licensed, as found in the LICENSE file.
