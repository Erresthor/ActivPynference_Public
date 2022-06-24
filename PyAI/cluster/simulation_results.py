#!/usr/bin/python
from json import load
import sys,inspect,os
import numpy as np
import matplotlib.pyplot as plt
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from pyai.base.file_toolbox import save_flexible,load_flexible
from pyai.layer.layer_learn import MemoryDecayType
from pyai.model.active_model import ActiveModel
from pyai.neurofeedback_run import evaluate_model_mean

from pyai.models_neurofeedback.climb_stairs import nf_model,evaluate_container

import time as t

# We grab a "performance_ouptut.pyai" file and extract the data :D
def load_perf(filepath):
    return (load_flexible(filepath))

if __name__=="__main__":
    savepath = os.path.join("C:",os.sep,"Users","annic","Desktop","Phd","code","results","series","series_a_b_prior")
    filename = "simulation_output.pyai"
    t0 = t.time()
    big_list = load_perf(os.path.join(savepath,filename))
    timefloat = (t.time()-t0)
    format_float = "{:.2f}".format(timefloat)
    print("Loaded performance file in " + format_float + " seconds.")
    
    t = np.arange(0,500,1)
    print(len(big_list))
    for model in big_list :
        model_object = model[0]
        performance_list = model[1]
        print(model_object.a)
        plt.plot(t,performance_list[6])
    plt.show()

    # We have a problem : these simulations are not ordered by design (we stack them up in a list :( )
    # Solution 1 : stack them up in an other form of data structure to preserve spatial coherence
    # SOlution 2 : sort them now depending on their variables
    # Solution 3 : include an "index" variable in ActiveModel class. (but we'd have to implement updates
    # of the model object between sessions.)




















