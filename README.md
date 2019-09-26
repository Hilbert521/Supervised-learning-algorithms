# Supervised Learning Algorithms for Controlling Underactuated Dynamical Systems
This repository contains Python implementation of the supervised learning algorithms devised in the paper
[Supervised Learning Algorithms for Controlling Underactuated Dynamical Systems](https://arxiv.org/abs/1909.11119)
by Bharat Monga and Jeff Moehlis. 

The paper details two related algorithms: Algorithm 1 and Algorithm 2, both of which output a binary control input by taking in the feedback of the state of the dynamical system. The algorithms learn this control input by sampling the state space and maximizing a reward function. Upon training, the algorithms generate two arrays X, U which contain the sampled states and the corresponding learnt control policies. These arrays together with the state x(t) of the system is fed to a classifier which outputs a binary control input. The working of the algorithms can be depicted as below:

<img src="https://github.com/bharatmonga/Supervised-learning-algorithms/blob/master/flowchart1.png" width="400">

The algorithms can be adapted to achieve various control objectives in several systems by engineering a suitable reward function. 

The folder [Algorithm 1](https://github.com/bharatmonga/Supervised-learning-algorithms/tree/master/Algorithm1) contains Python implementation of Algorithm 1 for a two dimensional duffing dynamical system. Here the objective is for the trajectory to converge to one of the stable fixed points of the system starting anywhere in the state space. Please have a look in the folder for more details.

The folder [Algorithm 2](https://github.com/bharatmonga/Supervised-learning-algorithms/tree/master/Algorithm2) contains Python implementation of Algorithm 2 for a three dimensional Lorenz dynamical system. Here the objective is for the trajectory to converge to the unstable fixed point of the system starting anywhere in the state space. Please have a look in the folder for more details.
