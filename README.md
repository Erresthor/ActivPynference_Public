# Active Pynference
Python package for simulating perception, decision-making & learning in uncertain environments using Active Inference (and more precisely Sophisticated Inference).

This package started off as a simple Python port of the Wellcome Centre for Human Neuroimaging's SPM12 "MDP_VB_XX" files but we're continuously working on additionnal orignal features ! Work in progress !


## Status

![status](https://img.shields.io/badge/status-active-green)
![Static Badge](https://img.shields.io/badge/python->=3.10-blue?logo=python)
![License](https://img.shields.io/badge/license-MIT-yellow)
![Accessibility](https://img.shields.io/badge/Accessible_on-TestPypi-orange?link=https%3A%2F%2Ftest.pypi.org%2Fproject%2Factive-pynference%2F)
![Publication](https://img.shields.io/badge/Published-No-red)


What you can do using <b>active-pynference</b> :
- Build your own Markov Decision Processes - based environments
- Build Active-Inference models of those environments
- <i>Link</i> those process and model together to simulate environments and various subject models structures
- Run simulations of agent behaviour & learning

What you cannot do (yet):
- Use the classical policy comparison scheme (SPM12's MDP_VB_X)
- Fit experimental data to find the optimal parameters
- Find [the answer to life the universe and everything](https://en.wikipedia.org/wiki/42_(number))


## Disclaimer !

This package is still under development ! Final testing and code cleaning are currently underway, be mindful of potential errors ! Contributions welcomed (feel free to write @ <come.annicchiarico@inserm.fr> !)

## Installation : 

In your Python environment, you (will soon be able to) install active-pynference using pip :
```
    pip install active-pynference
```

In the meanwhile, you can use the test version (be aware that there may be a plethora of bugs :/ ): 

```
pip install -i https://test.pypi.org/simple/ active-pynference
```

For more informations on package installation, check [the complete installation instructions notebook](demos/installation_instructions.ipynb).

## A general overview of the package design

## Examples 

- [Navigating a T-maze using Sophisticated Inference](demos/Tmaze_demo.ipynb)

A mouse has to pick either the left or the right (one-way) door. Behind one of the door is a reward (cheese), and behind the other is an adversive stimulus (mousetrap). The context may be either random or stable. The mouse also has access to a hint with an associated quality (i.e. *Does it actually provide any useful information about the reward whereabouts ?*). 

We use sophisticated inference to model the mouse learning both the context as well as the quality of the hint in various environments. Behavioural simulation outputs below : 


Reliable clue, random environment       | Reliable clue , environment stabilizes after trial 10 | Unreliable clue & random environment
:--------------------------------------:|:------------------------------------:|:------------------------------------:
![Image1](./resources/tmaze/renders/render_good_clue_2.gif) |![Image2](./resources/tmaze/renders/render_good_clue_cheese_stabilizes_at_10.gif)|![Image3](./resources/tmaze/renders/render_bad_clue_random_env.gif)



- [Navigating a complex "soft" maze](demos/mazeX_demo.ipynb)

An agent is taked with reaching a target cell in a complex "soft" maze. Contrary to a "hard" maze with uncrossable walls, there are adversive and neutral cells in this maze. An optimal agent tries to avoid adversive cells by exploring which cells are neutral and which cells are adversive. One interesting parameter we can play around in this simulation is the initial confidence of the agent regarding its prior mapping of the maze. When this confidence is very low, the agent will learn very fast. When it is too high, it won't learn at all. We can also toggle the novelty seeking part of the Sophisticated Inference planning algorithm to prompt more or less explorative behaviour.


<img src="demos/local_resources/mazex/renders/maze_explor_0.1.png" width="400">
<!-- ![Image1](demos/local_resources/mazex/renders/maze_explor_0.1.png) -->

<img src="demos/local_resources/mazex/renders/without_novelty_seeking.png" width="400">
<!-- ![Image2](demos/local_resources/mazex/renders/without_novelty_seeking.png) -->

<img src="demos/local_resources/mazex/renders/with_novelty_seeking.png" width="400">
<!-- ![Image3](demos/local_resources/mazex/renders/with_novelty_seeking.png) -->


## What is Active Inference ? Sophisticated Inference ?

Get in touch with the Active Inference ecosystem : 
- A nice syllabus : https://jaredtumiel.github.io/blog/2020/10/14/spinning-up-in-ai.html
- The original Matlab implementation of Sophisticated Inference : https://github.com/spm/spm/blob/main/toolbox/DEM/spm_MDP_VB_XX.m
- The original Sophisticated Inference paper : Karl Friston, Lancelot Da Costa, Danijar Hafner, Casper Hesp, Thomas Parr; Sophisticated Inference. Neural Comput 2021; 33 (3): 713–763. doi: https://doi.org/10.1162/neco_a_01351

## Paper / cite me !

We'll see about that later 	¯\\_(ツ)_/¯.