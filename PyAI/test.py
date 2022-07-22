from cmath import pi
import numpy as np
from pyai.base.matrix_functions import *
from pyai.base.function_toolbox import spm_wnorm


list_of_things = ["hey_1","hey_2","yala_2"]

my_thing = "yalo"

print(i for i in list_of_things)
print(my_thing in (i for i in list_of_things))

def filename_in_files(list_files,my_filename):
    for filename in (list_files):
        if my_filename in filename :
            return True
    return False

print(filename_in_files(list_of_things,my_thing))