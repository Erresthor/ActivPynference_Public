# Active Pynference
Python package for simulating perception, decision-making & learning in uncertain environments using Active Inference (and more precisely Sophisticated Inference).

This package started off as a simple Python port of the Wellcome Centre for Human Neuroimaging's SPM12 "MDP_VB_XX" files but we're continuously working on additionnal orignal features ! Work in progress !


## Status

![status](https://img.shields.io/badge/status-active-green)
![Static Badge](https://img.shields.io/badge/python-3.10-blue?logo=python)
![License](https://img.shields.io/badge/license-MIT-yellow)
![Accessibility](https://img.shields.io/badge/Accesible_on-TestPypi-orange?link=https%3A%2F%2Ftest.pypi.org%2Fproject%2Factive-pynference%2F)
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