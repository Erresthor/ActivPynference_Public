import numpy as np
import matplotlib.pyplot as plt
from pyai.base.normal_distribution_matrix import generate_normal_dist_along_matrix
from pyai.base.plotting_toolbox import multi_matrix_plot
from pyai.base.function_toolbox import normalize

if __name__=="__main__":
    matrix = (10*generate_normal_dist_along_matrix(np.eye(5),0.05)+1)
    matrix = normalize(matrix)
    multi_matrix_plot([matrix,matrix],["A matrix","the same"])
    plt.show()