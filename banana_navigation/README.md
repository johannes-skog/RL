#### Environment to solve

Collect as many yellow bananas as possible without collecting any blue bananas.

Collecting a yellow banana yeilds +1 in reward, collecting a blue banana yeilds -1.

One episode consists of 300 actions

![Environment](img/banana.gif)

Four different actions are possible to takes:

  ← ↑ ↓ →

The state consists of 37 values mapping to the velocity of the agent and values corresponding to ray-based perception of objects in the forward direction of the agent. An example state would be:

     [1.         0.         0.         0.         0.84408134 0.
     0.         1.         0.         0.0748472  0.         1.
     0.         0.         0.25755    1.         0.         0.
     0.         0.74177343 0.         1.         0.         0.
     0.25854847 0.         0.         1.         0.         0.09355672
     0.         1.         0.         0.         0.31969345 0.
     0.]


The enviroment can be regarded as solved if the agent is able to get an average total reward of > 13 over 100 episodes.

#### Dependencies

Linux x86-64 system

Install Anaconda python 3, [download](https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh)

For setting up the conda environment and download the unity environment run

    bash setup.bash

Make sure to activate the conda environment

    conda activate banana_navigation

#### Jupyter Notebook & Tensorboard

In order to analyze/visualize how different choices of hyperparameters effecs the agent's ability to solve the enviroment, launch the notebook and tensorboard service by running

    bash launch.bash

![Environment](img/tensorboard.png)

#### Train the agent

Make sure that all dependencies have been installed and that the notebook and tensorboard service have been started. Go into the analyze notebook and follow the instructions there.

#### Agent description and conclusions

In the report notebook you will find a description of the implemented agent together with conclusions and future work.
