# Supervised Learning Algorithms for Controlling Underactuated Dynamical Systems
This repository contains Python implementation of the supervised learning algorithms devised in the paper
[Supervised Learning Algorithms for Controlling Underactuated Dynamical Systems](https://arxiv.org/abs/1909.11119)
by Bharat Monga and Jeff Moehlis. 

The paper details two related algorithms: Algorithm 1 and Algorithm 2, both of which output a bang-bang (binary) control input by taking in feedback of the state of the dynamical system. The algorithms learn this control input by maximizing a reward function in both short and long time horizons. Upon training, the algorithms generate two arrays X, U which contain the sampled states and the corresponding control policies. This together with the state of the system is fed to a classifier which outputs a binary control. The working of the algorithms can be depicted as below:

<img src="https://github.com/bharatmonga/Supervised-learning-algorithms/blob/master/flowchart1.pdf" width="700">
