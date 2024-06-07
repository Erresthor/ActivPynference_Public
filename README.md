# Active Pynference
Python package for simulating perception, decision-making & learning in uncertain environments using Active Inference (and more precisely Sophisticated Inference).

This package started off as a simple Python port of the Wellcome Centre for Human Neuroimaging's SPM12 "MDP_VB_XX" files but we're continuously working on additionnal orignal features ! Work in progress !

What you can do using <b>active-pynference</b> :
- Build your own Markov Decision Processes - based environments
- Build Active-Inference models of those environments
- <i>Link</i> those process and model together to simulate environments and various subject models structures
- Run simulations of agent behaviour & learning

What you cannot do (yet):
- Use the classical policy comparison scheme (SPM12's MDP_VB_X)
- Fit experimental data to find the optimal parameters
- Find [the answer to life the universe and everything](https://en.wikipedia.org/wiki/42_(number))

## Installation : 

In your Python environment, you may install active-pynference using pip :
```
    pip install active-pynference
```

For more informations on package installation, check [the complete installation instructions notebook](demos/installation_instructions.ipynb).

## A general overview of the package design

## Examples 

- [Navigating a T-maze using Sophisticated Inference](demos/T-maze_demo.ipynb)

Behavioural simulation outputs : 

![Image1](./resources/tmaze/renders/render_good_clue_2.gif)

![Image2](./resources/tmaze/renders/render_good_clue_cheese_stabilizes_at_10.gif)

![Image3](./resources/tmaze/renders/render_bad_clue_random_env.gif)

- [Navigating a complex "soft" maze](demos/mazeX_demo.ipynb)

## What is Active Inference ? Sophisticated Inference ?

Get in touch with the Active Inference ecosystem : 
- A nice syllabus : https://jaredtumiel.github.io/blog/2020/10/14/spinning-up-in-ai.html
- The original Matlab implementation of Sophisticated Inference : https://github.com/spm/spm/blob/main/toolbox/DEM/spm_MDP_VB_XX.m
- The original Sophisticated Inference paper : Karl Friston, Lancelot Da Costa, Danijar Hafner, Casper Hesp, Thomas Parr; Sophisticated Inference. Neural Comput 2021; 33 (3): 713–763. doi: https://doi.org/10.1162/neco_a_01351

## Paper / cite me !

We'll see about that later 	¯\\_(ツ)_/¯.