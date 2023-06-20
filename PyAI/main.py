from demo.new_layer_test import run_test
from demo.link_test import linktest
# from pyai.architecture.layer_link import get_negative_range,transmit_data
# from pyai.architecture.layer_link import get_margin_distribution_along,get_joint_distribution_along
# from pyai.base.function_toolbox import spm_cross,normalize
# from pyai.base.miscellaneous_toolbox import isField
import numpy as np

class basic_object:
    def __init__(self):
        self.o_d = np.array([[0.2,0.3,0.3],
                             [0.2,0  ,  0]])
        
        self.x_d = np.array([[0.1,0.2],
                             [0.5,0.1],
                             [0.1,0  ]])

    def __str__(self):
        return_this ="-----------------------\n"
        return_this +='o_d\n'
        return_this +=str(np.round(self.o_d,2))
        return_this +='\nx_d\n'
        return_this +=str(np.round(self.x_d,2))
        return_this +="\n-----------------------\n"
        return return_this

if __name__ == '__main__':
    linktest()