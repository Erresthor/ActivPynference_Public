# import seaborn as sns
# sns.set(color_codes=True)
from scipy.stats import uniform,norm
from scipy.stats import gamma,cauchy,expon,rdist
import pandas as pd
from audioop import avg
from unicodedata import unidata_version
import numpy as np
import matplotlib.pyplot as plt
import pandas
from matplotlib.animation import FuncAnimation
from pyai.base.matrix_functions import matrix_distance_list
from pyai.base.function_toolbox import normalize,spm_KL_dir,KL_test,KL_div_variant
from pyai.base.matrix_functions import multidimensionnal_uncertainty


A = np.array([[0.5,0.3,0.2],
              [0.0,0.3,0.7],
              [0.0,0.0,1.0]])

K = np.array([0,0,0.1,0.3,0.4,0.15,0.05,0.0])

def avg_dist_entropy(matrix,eps=1e-8,MAXIMUM_INFO = 10000):
    """ Calculates the average of the marginal distributions entropy (dist are along axis 0)"""
    matrix = normalize(matrix)
    
    zeromatrix = np.zeros(matrix.shape)
    zeromatrix[matrix<eps] = np.ones(matrix.shape)[matrix<eps]*MAXIMUM_INFO
    
    zeromatrix[matrix>=eps] = zeromatrix[matrix>=eps] - np.log(matrix[matrix >= eps])

    marginal_entropy = zeromatrix*matrix
    matrix_entropy = np.sum(marginal_entropy,axis = 0)

    return (np.average(matrix_entropy))

def flexible_entropy(matrix_or_list,norm=True):
    if (type(matrix_or_list)==list):
        out = []
        for k in range(len(matrix_or_list)):
            out.append(flexible_entropy(matrix_or_list[k]))
        return out
    elif (type(matrix_or_list)==np.ndarray):
        normalizer = 1
        if (norm):
            normalizer = np.log(matrix_or_list.shape[0])
            # The entropy of a uniform distribution of the same size
        return avg_dist_entropy(matrix_or_list)/normalizer
    else :
        return 0

def arrays(data,Nx = 100):
    N = data.shape[0]
    X = np.linspace(0,10,Nx+1)
    Y = np.zeros((Nx+1,))
    for K in range(N) :
        y = data[K]
        for i in range(Nx):
            xi = X[i]
            xiplus = X[i+1]
            if(y>=xi)and(y<xiplus):
                Y[i] += 1
    return X[:-1],Y[:-1]      

avg_dist_entropy(normalize(A))
avg_dist_entropy(normalize(K))
N = 10000
start = 0
end = 10

def plot_uncertainty_estimlators():

    data_uniform = uniform.rvs(size=N,loc=start,scale=end)

    data_norm = norm.rvs(size=N,loc=(start+end/2),scale=1)

    data_norm_narrow = norm.rvs(size=N,loc=(start+end/2),scale=0.5)

    data_gamma = gamma.rvs(a=5, size=N)

    # data_cauchy = cauchy.rvs(size = N,loc = (start+end/2),scale=0.35)

    # data_expon = expon.rvs(size = N,loc = 0,scale=1)

    # data_r = rdist.rvs(size = N,loc =(start+end/2),c=4.9,scale=1.0)



    Nx = 200

    X,Yuni = arrays(data_uniform,Nx)
    X,Ynor = arrays(data_norm,Nx)
    X,Ygam = arrays(data_gamma,Nx)
    X,Ynornar = arrays(data_norm_narrow,Nx)
    # X,Ycauchy = arrays(data_cauchy,Nx)
    # X,Yexp = arrays(data_expon,Nx)
    # X,Yr = arrays(data_r,Nx)
    Yr = np.zeros((Nx,))
    Yr[:int(Nx/3.0)] = 500
    Yr[int(Nx/3.0):int(2*Nx/3.0)] = 10
    Yr[int(2*Nx/3.0):] = 50

    Dist_uni = normalize(Yuni)
    Dist_nor = normalize(Ynor)
    Dist_gam = normalize(Ygam)
    Dist_nar = normalize(Ynornar)
    # Dist_cauchy = normalize(Ycauchy)
    # Dist_exp = normalize(Yexp)
    Dist_r = normalize(Yr)

    ordered_uni = -np.sort(-Dist_uni)
    ordered_nor = -np.sort(-Dist_nor)
    ordered_gam = -np.sort(-Dist_gam)
    ordered_nar = -np.sort(-Dist_nar)
    # ordered_cauchy = -np.sort(-Dist_cauchy)
    # ordered_exp = -np.sort(-Dist_exp)
    ordered_r = -np.sort(-Dist_r)

    def uncertainty(distribution,factor = 1/2):
        """ For factor in [0,1[, a measure of how "dispersed" the information is in subject beliefs."""
        
        K = distribution.shape[0]
        print(K)
        normalized_distribution = normalize(distribution)
        sorted_dist = -np.sort(-normalized_distribution)

        def find_point_in(y,data):
            """ data is an decreasing set of probability density"""
            x = -1
            K = data.shape[0]
            for k in range (K-1):
                x += 1
                if (data[x]>y) and (data[x+1]<=y):
                    return(x)
            return K

        def uncertainty_estimator(distribution,epsilon= 0):
            """ Which proportion of the distribution space accounts for probabilities > 1/N - epsilon ?"""
            N = distribution.shape[0]
            fixed_point_y = min(1/N - epsilon,N)
            assert fixed_point_y >= 0, "uncertainty parameter should be > 1 / distribution space size, instead of " + str(fixed_point_y)

            fixed_point_x = find_point_in(fixed_point_y,distribution)
            return fixed_point_x/N,fixed_point_x,fixed_point_y
        
        epsilon = factor*(1.0/K)
        rating,x,y =  uncertainty_estimator(sorted_dist,epsilon)
        return rating

    # epsilon = 0.005
    # print(uncertainty_estimator(ordered_uni,epsilon))
    # print(uncertainty_estimator(ordered_nor,epsilon))
    # print(uncertainty_estimator(ordered_nar,epsilon))
    # print(uncertainty_estimator(ordered_gam,epsilon))
    # fixed_point = (1/K) - epsilon
    # print(find_point_in(fixed_point,ordered_uni))
    # print(find_point_in(fixed_point,ordered_nor))
    # print(find_point_in(fixed_point,ordered_gam))
    K = Yuni.shape[0]
    X = np.linspace(0,K,K)

    plt.plot(X,Dist_uni,color='red',label='uniform')
    plt.plot(X,Dist_nor,color='blue',label='norm')
    plt.plot(X,Dist_nar,color='cyan',label='norm (sig = 0.5)')
    plt.plot(X,Dist_gam,color='green',label='gamma')
    # plt.plot(X,Dist_cauchy,color='purple',label='cauchy')
    # plt.plot(X,Dist_exp,color='yellow',label='exponential')
    plt.plot(X,Dist_r,color='brown',label='rdist')
    plt.legend()
    plt.show()

    # plt.plot(X,Dist_uni,color='red',label='uniform')
    # plt.plot(X,Dist_nor,color='blue',label='norm')

    # plt.plot(X,Dist_gam,color='green',label='gamma')
    # plt.legend()
    # plt.show()


    fac = 0.5
    xuni = uncertainty(Yuni,fac)*K
    xnor = uncertainty(Ynor,fac)*K
    xnar = uncertainty(Ynornar,fac)*K
    xgam = uncertainty(Ygam,fac)*K
    # xcau = uncertainty(Ycauchy,fac)*K
    # xexp = uncertainty(Yexp,fac)*K
    xr = uncertainty(Yr,fac)*K


    huni = avg_dist_entropy(Yuni)
    hnor = avg_dist_entropy(Ynor)
    hnar = avg_dist_entropy(Ynornar)
    hgam = avg_dist_entropy(Ygam)
    hr = avg_dist_entropy(Yr)

    #Entropy between 0 (spike) and 1 (uniform):
    normalizing_contant = np.log(Nx)
    hnor = hnor/normalizing_contant
    hnar = hnar/normalizing_contant
    hgam = hgam/normalizing_contant
    hr = hr/normalizing_contant
    huni = huni/normalizing_contant

    #Metric entropy
    # hnor = hnor/Nx
    # hnar = hnar/Nx
    # hgam = hgam/Nx
    # hr = hr/Nx
    # huni = huni/Nx


    lw = 2

    ytarg = 1/K - fac/K
    plt.axhline(y=ytarg, color='black', linestyle='--')

    plt.plot(X,ordered_uni,color='red',label='uniform',linewidth=lw)
    plt.axvline(x=xuni, color='red', linestyle=':')
    plt.axvline(x=huni, color='red', linestyle='--')

    plt.plot(X,ordered_nor,color='blue',label='norm,sig = 1',linewidth=lw)
    plt.axvline(x=xnor, color='blue', linestyle=':')
    plt.axvline(x=hnor, color='blue', linestyle='--')

    plt.plot(X,ordered_nar,color='c',label='norm,sig = .5',linewidth=lw)
    plt.axvline(x=xnar, color='c', linestyle=':')
    plt.axvline(x=hnar, color='c', linestyle='--')

    plt.plot(X,ordered_gam,color='green',label='gamma',linewidth=lw)
    plt.axvline(x=xgam, color='green', linestyle=':')
    plt.axvline(x=hgam, color='green', linestyle='--')

    # plt.plot(X,ordered_cauchy,color='purple',label='cauchy',linewidth=lw)
    # plt.axvline(x=xcau, color='purple', linestyle=':')

    # plt.plot(X,ordered_exp,color='yellow',label='exponential',linewidth=lw)
    # plt.axvline(x=xexp, color='yellow', linestyle=':')

    plt.plot(X,ordered_r,color='brown',label='rdist',linewidth=lw)
    plt.axvline(x=xr, color='brown', linestyle=':')
    plt.axvline(x=hr, color='brown', linestyle='--')

    #plt.ylim([-1e-3,0.01])
    plt.legend()
    plt.show()



    facl = np.linspace(0,1,100)
    xuni = []
    xnor = []
    xnar = []
    xgam = []
    xcau = []
    xexp = []
    xr = []

    for fac in facl:
        #facl.append(fac)
        xuni.append(uncertainty(Yuni,fac))
        xnor.append(uncertainty(Ynor,fac))
        xnar.append(uncertainty(Ynornar,fac))
        xgam.append(uncertainty(Ygam,fac))
        # xcau.append(uncertainty(Ycauchy,fac))
        # xexp.append(uncertainty(Yexp,fac))
        xr.append(uncertainty(Yr,fac))
    print(facl,xuni)

    plt.plot(facl,xuni,color='red',label='uniform')
    plt.axhline(y=huni, color='red', linestyle='--')

    plt.plot(facl,xnor,color='blue',label='norm')
    plt.axhline(y=hnor, color='blue', linestyle='--')

    plt.plot(facl,xnar,color='c',label='norm sig=.5')
    plt.axhline(y=hnar, color='c', linestyle='--')

    plt.plot(facl,xgam,color='g',label='gamma')
    plt.axhline(y=hgam, color='green', linestyle='--')

    # plt.plot(facl,xcau,color='purple',label='cauchy')
    # plt.plot(facl,xexp,color='yellow',label='exponential')

    plt.plot(facl,xr,color='brown',label='rdist')
    plt.axhline(y=hr, color='brown', linestyle='--')
    plt.xlabel("Factor ")
    plt.ylabel("Measure of uncertainty/dispersion")
    plt.title("Value of dispersion depending on factor for different distributions")
    plt.legend()
    plt.show()


if __name__=="__main__":
    number_of_ticks = np.logspace(0,6,150)
    K = []
    approxer = []
    for nt in number_of_ticks:
        print(nt,int(nt))
        uniform = np.ones((int(nt),))/int(nt)
        K.append(avg_dist_entropy(uniform))
        approxer.append(np.log(nt))

    plt.plot(number_of_ticks,approxer,color='r')
    plt.plot(number_of_ticks,K)

    # convert y-axis to Logarithmic scale
    plt.xscale("log")
    
    plt.show()

    #plot_uncertainty_estimlators()