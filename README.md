# Information Directed Sampling on Multi Arm Bandit problems

Implementation of Russo and Van Roy (2016)

## Simulate a Multi-Arm bandit

The arms.py file provides the bricks to add arms with different probability laws for the reward. The MAB.py file builds a multi-arm bandit given 
the arms as input and provide the Upper-Confidence-Bound (UCB) and Thompson Sampling (TS) to find the best arm.

## Linear Bandit

LinearMAB.py contains a class to implement a Linear Bandit problem and the LinUCB algorithm to minimize the regret and find the best feature vector.

