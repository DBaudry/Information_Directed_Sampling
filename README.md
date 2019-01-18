# Information Directed Sampling for Multi-Arm Bandit problems

Our work is based on an article from Russo & Van Roy (2018): [Learning to optimize via Information Directed Sampling](https://github.com/nchopin/particles) .
The aim of this project is to propose a platform that allows to reproduce the experiments of the paper and to easily implement new examples.
We first implement classes to implement a general Multi-Arm Bandit setting. Then, we provide some problem specific classes that implements different versions of IDS and some widely-studied algorithms to compare the performance IDS with the results of these algorithms.
The aim of these algorithm is to minimize the Bayesian Regret for the specified problem.

## General settings: MAB

In the MAB file we implement the generic class for a Multi Arm Bandit settings. We implemented the algorithms that are not problem specific and some useful function to handle the results of the algorithm.

## Specific problem classes

We provide a few classes we settings and functions to run experiments for the problem:

* FiniteSets: A first algorithm for the case where the parameter space, action space and outcome space are all finites
* BernoulliMAB: independent Bernoulli arms with parameters drawn uniformly at random. 
* GaussianMAB: independent Gaussian arms where the mean is drawn at random with the gaussian distribution
* LinearMAB: Linear Gaussian bandit with independent features, this class does not use MAB but a specific class designed for Linear Bandits. All parameters are drawn from Gaussians in this model

## How to run Experiment

We provide an expe file with functions that help to run experiments for the pre-cited problems easily. These function are called in the main file, which is designed to help the choice of the arguments in the expe functions.
