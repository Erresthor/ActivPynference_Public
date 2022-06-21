import numpy as np
from .function_toolbox import normalize
import matplotlib.pyplot as plt
import math
import pandas as pd

def normal_pdf(mu,sigma_2,x):
    return (1.0/(np.sqrt(sigma_2*2*math.pi)))*np.exp(-0.5*np.power((x-mu),2)/sigma_2)

def cast_to_array(x,y,scale_array):
    # x is the x axis of the distribution --> Should be ordered in ascending fashion
    # y is the local p(x)
    # array is the x scale to be cast upon (array) --> Should be ordered in ascending fashion
    # We round the normal distribution to the closest value of scale_array
    normal_casted = np.zeros(scale_array.shape)
    scale_array_counter = 0

    integral = 0
    total_points = x.shape[0]

    for xi in range(total_points):

        if ((xi<total_points-1)and(xi>0)): # If we are the first or last point, ignore it, else :
            current_scale_array = scale_array[scale_array_counter]
            try :
                upper_bound = scale_array[scale_array_counter + 1]
                limit = (current_scale_array + upper_bound)/2.0 # limit : if I am below, I should be counted towards the current scale element
                                                            # If I am above, I no longer counts towards this scale element and counter should 
                                                            # Increase !
            except :
                limit = np.inf

            if (x[xi] > limit): # If the point we analyze is above the limit, it should belong to the next scale array element
                                # We save the current integral to the scale element, and then change element
                normal_casted[scale_array_counter] = integral
                scale_array_counter += 1

                distance_for_cast = x[xi]-x[xi-1]
                integral = y[xi]*distance_for_cast # We neglect a part of the distro under the limit, not important if scale(x) <<< scale(scale_array)
            else : # This slice belongs to the current scale element, we add this slice of the distro to the integral
                distance_for_cast = x[xi]-x[xi-1] 
                integral += y[xi]*distance_for_cast        
        elif (xi==total_points-1) :
            normal_casted[scale_array_counter] = integral
            scale_array_counter += 1
    
    return normal_casted

# x = np.linspace(0,10,1000)
# y = normal_pdf(3,0.3,x)
# cast_to = np.array([1,2,3,4,5,6,7,8,9])
# #cast_to = np.array([0.5,4.5,7.,7.8,7.9])
# casted = cast_to_array(x,y,cast_to)
# print(casted,np.sum(casted))

# print(np.linspace(0,10,10+1))
def generate_distribution(empty_array,mu,sig2,n_points = 500,ecart = 3) :
    assert empty_array.ndim ==1,"Array should be 1 dimensionnal"
    k = empty_array.shape[0]

    ecart = max(ecart,0.5)
    x = np.linspace(0-ecart,k-1+ecart,n_points)
    # Distribution to approximate the normal distribution
    y = normal_pdf(mu,sig2,x)
    # Corresponding normal distribution

    # The space to project it unto : [0 -> matrix_line-1]
    X = np.linspace(0,k-1,k).astype(int)
    casted_dist = cast_to_array(x,y,X)
    return X,casted_dist,x,y

def generate_normal_dist_along_matrix(mu_matrix,sig2,n_points=500,ecart=3) :
    #Along axis 0. The mu value used is the mean of the matrix along axis 0 :
    mus = np.argmax(mu_matrix,axis=0)

    output_matrix = np.zeros(mu_matrix.shape)
    it = np.nditer(mus, flags=['multi_index'])
    for i in (it):
        slicer = tuple([slice(None)]) + it.multi_index
        X,casted_dist,x,y = generate_distribution(np.zeros(mu_matrix[slicer].shape),i,sig2,n_points,ecart)
        output_matrix[slicer] = casted_dist
    return output_matrix

def generate_normal_dist_along_mulist(zero_matrix,mulist,sig2,n_points=500,ecart=3) :
    #Along axis 0. The mu value used is the mean of the matrix along axis 0 :
    assert mulist.shape == zero_matrix[0,...].shape, "Mu array shape should be " + str( zero_matrix[0,...].shape) + " but is " + str(mulist.shape)

    output_matrix = zero_matrix
    it = np.nditer(mulist, flags=['multi_index'])
    for i in (it):
        slicer = tuple([slice(None)]) + it.multi_index
        X,casted_dist,x,y = generate_distribution(np.zeros(zero_matrix[slicer].shape),i,sig2,n_points,ecart)
        output_matrix[slicer] = casted_dist
    return output_matrix

# X,casted_dist,x,y = generate_distribution(np.zeros(5,),4.2,1,ecart=10)
# plt.bar(X,casted_dist,0.95)
# # plt.plot(x,y,color='r')
# plt.show()

if(__name__=="__main__") :
    from pyai.base.plotting_toolbox import multi_3dmatrix_plot,multi_matrix_plot
    mats  =[]
    labs = []

    # Mu PERFECT

    sigmas = np.array([0.01,0.1,0.25,0.5,1,3,10,50])
    s_size = sigmas.shape[0]

    ec = 0
    npoints = 1000
    k = 5

    A = np.eye(k)
    As = np.zeros(A.shape+(s_size,))
    for ss in range(s_size):
        sig = sigmas[ss]
        a = generate_normal_dist_along_matrix(A,sig,n_points=npoints,ecart=ec)
        print(a.shape,As.shape)
        As[...,ss] = a

    mats.append(As)
    labels = (np.core.defchararray.add("mu = real, Sigma² = ", sigmas.astype(str)))
    labs.append(labels)

    # Mu NOT PERFECT
    perfect = np.linspace(0,k-1,k)
    mulist = perfect*0.8
    Bs = np.zeros(A.shape+(s_size,))
    for ss in range(s_size):
        sig = sigmas[ss]
        b = generate_normal_dist_along_mulist(np.zeros(A.shape),mulist,sig,n_points=npoints,ecart=ec)
        Bs[...,ss] = b
    mats.append(Bs)
    labels = (np.core.defchararray.add("mu = real x 0.8, Sigma² = ", sigmas.astype(str)))
    labs.append(labels)

    # Mu NOT PERFECT
    perfect = np.linspace(0,k-1,k)
    mulist = perfect+0.5
    Cs = np.zeros(A.shape+(s_size,))
    for ss in range(s_size):
        sig = sigmas[ss]
        b = generate_normal_dist_along_mulist(np.zeros(A.shape),mulist,sig,n_points=npoints,ecart=ec)
        Cs[...,ss] = b
    mats.append(Cs)
    labels = (np.core.defchararray.add("mu = real + 0.5 Sigma² = ", sigmas.astype(str)))
    labs.append(labels)

    multi_matrix_plot(mats,labs,colmap = 'jet',xlab="Hidden mental state",ylab="Observation")
    plt.show()