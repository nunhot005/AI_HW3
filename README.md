# AI_HW3
local search algorithm  
ITCS451_Artificial Intelligence  

## Description:

First, you need to install two packages: OpenAI's gym and Numpy.

In this assignment, you will implement 2 functions related to the local search algorithms to find an agent that solves Cart-Pole game. In the template file itcs451-hw2.py, you will see "# TODO: 1" and "# TODO: 2". You can read the documentation for what you need to do in detail. Here is a summary:

1. hillclimb_sideway(): Hill-Climbing search with sideway trick.

2. simulated_annealing(): Simulated Annealing search.

## There are 2 ways to run this code. 

1. If you want to search, you can simply run
python itcs451-hw3.py
Make sure that you change your search method in the main section of the code.
2. If you want to see how your agent performs, you can run
python itcs451-hw3.py "1, 0.9, 2, 2, 1"
, where the five numbers are your agent or 
python itcs451-hw3.py "random"
, for a random agent.
