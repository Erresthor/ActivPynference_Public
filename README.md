# Active Pynference
Python package for simulating perception, decision-making & learning in uncertain environments using Active Inference (and more precisely Sophisticated Inference).

This package started off as a simple Python port of the Wellcome Centre for Human Neuroimaging's SPM12 "MDP_VB_XX" files but we're continuously working on additionnal orignal features ! Work in progress !

What you can do using <b>active-pynference</b> :
- Build your own Active Inference structures (model & process) using a generic <i> layer </i> component
- <i>Link</i> those structures together to simulate environments and various subject models of those environments
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

For more informations on package installation, check [the complete installation instructions notebook](READUS/installation_instructions.ipynb).

## Active Inference ?

## A general overview of the package design

## Examples 

- [Navigating a T-maze using Sophisticated Inference](READUS/T-maze_demo.ipynb)

![Image1](.\resources\tmaze\renders\render_good_clue_2.gif)

![Image2](.\resources\tmaze\renders\render_good_clue_cheese_stabilizes_at_10.gif)

![Image3](.\resources\tmaze\renders\render_bad_clue_random_env.gif)

- [Navigating a complex "soft" maze](READUS/mazeX_demo.ipynb)

## Paper / cite me !

We'll see about that later 	¯\\_(ツ)_/¯.