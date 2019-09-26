# Algorithm 1
This repository contains Python implementation of Algorithm 1 for a two dimensional Duffing dynamical system. Here the objective is for the trajectory to converge to the stable fixed point [1, 0] of the system starting anywhere in the state space.

[duffing.py](https://github.com/bharatmonga/Supervised-learning-algorithms/blob/master/Algorithm1/duffing.py) creates a Duffing class which can be used to initiate a duffing object. It has attributes like reseting the state of the object, a classifier that outputs binary control, a step function that returns the state at the next time step, a function to return reward associated with the current state of the object, and a function to generate a trajectory with or without any control.

Run [train.py](https://github.com/bharatmonga/Supervised-learning-algorithms/blob/master/Algorithm1/train.py) to generate the training data X and U and store them in a file. The arrays X, and U contain the sampled states and the corresponding learnt control policies, which can be visualized as below: <img src="https://github.com/bharatmonga/Supervised-learning-algorithms/blob/master/Algorithm1/Algorithm1_training_data_visualization.png" width="600">

where the blue circles (resp., black x's) represent elements of X where the control policy given by elements of U is OFF (resp., ON). The green (resp., red) region is where the output of the binary classifier is OFF (resp., ON).

After running [train.py](https://github.com/bharatmonga/Supervised-learning-algorithms/blob/master/Algorithm1/train.py) file, run [compute_trajectory.py](https://github.com/bharatmonga/Supervised-learning-algorithms/blob/master/Algorithm1/compute_trajectory.py), which will load up training data from the file previosly saved and use that to generate a controlled trajectory. The program will also generate an uncontrolled trajectory for comparison. The trajectories are visualized as above.
<img src="https://github.com/bharatmonga/Supervised-learning-algorithms/blob/master/Algorithm1/duffing_trajectory.png" width="600">

After running [train.py](https://github.com/bharatmonga/Supervised-learning-algorithms/blob/master/Algorithm1/train.py) file, run [validation.py](https://github.com/bharatmonga/Supervised-learning-algorithms/blob/master/Algorithm1/validation.py), which tests the effectiveness of the learning algorithm. It loads up training data from the file previously saved and generates 1000 controlled trajectories from different initial conditions. It prints effectiveness of the algorithm by analyzing end states of the 1000 trajectories to see if they are close to the desired state or not. The start and end states of the 1000 trajectories is visualized as below:

<img src="https://github.com/bharatmonga/Supervised-learning-algorithms/blob/master/Algorithm1/learning_algorithm1_efficiency.png" width="600">

As you can see from the above figure, the algorithm achieves 100% effectiveness as the final states of all the trajectories lie very close to the desired state of the Duffing dynamical system.
