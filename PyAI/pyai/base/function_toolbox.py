# -*- coding: utf-8 -*-
"""
Created on Fri May  7 11:58:11 2021

@author: cjsan
"""
import numpy as np
from scipy.special import gammaln
from scipy.special import psi
import random as random


def precision_weight(A,gamma,center = False):
    if (type(A)==list):
        result = []
        for i in range(len(A)):
            if ((type(gamma)==np.ndarray)or(type(gamma)==list)):
                result.append(precision_weight(A[i],gamma[i],center=center))
            else :
                result.append(precision_weight(A[i],gamma,center=center))
        return result
    else :
        return softmax(A**gamma,0,center)



def softmax_appr(X):                                                                 ###converts log probabilities to probabilities
      norm = np.sum(np.exp(X)+10**-5)
      Y = (np.exp(X)+10**-5)/norm
      return Y

def softmax(X,axis = None,center=True):
    if not(axis==None) :
        if center : 
            x = np.exp(X - np.max(X)) #,axis=axis))
        else :
            x = np.exp(X)
        return x/np.sum(x,axis=axis,keepdims=True)
    else:
        if center :
            x = np.exp(X - np.max(X))
        else :
            x = np.exp(X)
        Y = x/np.sum(x)
    return Y

def softmax_dim2(X):                                                            ###converts matrix of log probabilities to matrix of probabilities
    norm = np.sum(np.exp(X)+10**-5,axis=0)
    Y = (np.exp(X)+10**-5)/norm
    return Y

def normalize(X,axis=0,epsilon=1e-15):                                                               ###normalises a matrix of probabilities
    if(type(X) == list):
        x =[]
        for i in range(len(X)):
            x.append(normalize(X[i],axis=axis))
    
    elif (type(X)==np.ndarray) :
        X = X.astype(float)

        X[X<0] = 0

        Xint =  np.sum(X,axis,keepdims=True) < epsilon
        Xint = np.repeat(Xint,X.shape[axis],axis=axis)
        X[Xint] = X[Xint] + epsilon
        x= X/(np.sum(X,axis))
    elif(X==None):
        return None
    else :
        print("Unknwon type encountered in normalize : " + str(type(X)))
        print(X)
    return x

def nat_log(x):
    #return np.log(x+np.exp(-16))
    return np.log(x+1e-16)

def md_dot(A,s,f):
    """ Dot product A.s along dimension f"""
    if f==0 :
        B = np.dot(A.T,s)
    elif f ==1 :
        B = np.dot(A,s)
    else :
        return None
    return B

def spm_wnorm_toocomplex(A):
    # no idea what it does ¯\_(ツ)_/¯ 
    tosqueeze = False
    if (A.ndim < 2):
        A = np.expand_dims(A,axis=0)
        tosqueeze=True
    A = A + 1e-16
    A_wnormed = ((1./np.sum(A,axis=0,keepdims=True)) - (1./A))/2.
    if tosqueeze : 
        return (np.squeeze(A_wnormed))
    return A_wnormed

def spm_wnorm(A) :
        # no idea what it does ¯\_(ツ)_/¯ 
    A = A + 1e-16 #♣np.exp(-16)
    A_wnormed = ((1./np.sum(A,axis=0,keepdims=True)) - (1./A))/2.
    return np.squeeze(A_wnormed)

def spm_cross(*argv) :
    n = (len(argv))
    if n < 2 :
        arg = argv[0]
        if (not((type(arg)==list) or (type(arg)==tuple))) :
            return arg
        else :
            newarg = tuple(arg)
            return spm_cross(*newarg)

    arg1 = argv[0]
    n1 = arg1.ndim
    
    arg2 = argv[1]
    n2 = arg2.ndim
    
    A = np.reshape(arg1,(arg1.shape + tuple([1 for x in range(n2)])))
    B = np.reshape(arg2,(tuple([1 for x in range(n1)]) + arg2.shape)) 
    Y = np.squeeze(A*B)
    
    if (n>2):
        newarg = (Y,) + argv[2:]
        return spm_cross(*newarg)
    else :
        return Y

def cell_md_dot(X,x) : 
    """x being a list of arrays"""
    #% initialize dimensions
#    DIM = (1:numel(x)) + ndims(X) - numel(x);
    DIM = []
    for k in range(len(x)):
        DIM.append(k + X.ndim - len(x))
#    # compute dot product using recursive sums (and bsxfun)
#    for d = 1:numel(x)
    for f in range(len(x)):
        s = np.ones((X.ndim,))
        s[DIM[f]] = len(x[f])
        X = X*np.reshape(x[f],tuple(s.astype(np.int)))   
        X = np.sum(X,axis = DIM[f],keepdims=True)
    return np.squeeze(X)

def G_epistemic_value(A,s):
    def episte_log(x):
        return np.log(x + np.exp(-16))
    
    qx = spm_cross(s) 
    G = 0
    qo = 0
    for i in np.argwhere(qx>np.exp(-16)) :
        po = np.array([1.])
        for elA in range(len(A)) :
            tup = tuple(np.concatenate(([...],i)))
            po = spm_cross(po,A[elA][tup])
        po = po.ravel()
        qo = qo + qx[tuple(i)]*po
        G = G + qx[tuple(i)]*np.dot(po.T,episte_log(po))
    G = G - np.dot(qo.T,episte_log(qo))
    return G

def KL_test(X,Y) :
    """KL divergence """
    X = normalize(X)
    Y = normalize(Y)
    assert X.ndim == Y.ndim , "Number of dimensions don't match ! " + str(X.ndim) + " vs " + str(Y.ndim) + " ."
    assert X.shape == Y.shape , "Dimensions of array to compare don't match ! " + str(X.shape) + " vs " + str(Y.shape) + " ."
    return(np.sum(X * (nat_log(X)-nat_log(Y))))

def KL_div_variant(x,y,norm=False):
    if (norm):
        x = normalize(x)
        y = normalize(y)
    D = (gammaln(np.sum(x)) - gammaln(np.sum(y)) - np.sum(gammaln(x)) + 
            np.sum(gammaln(y)) + np.dot((x-y).T,(psi(x)-psi(np.sum(y)))))
    return np.sum(D)     

def spm_KL_dir(x1,x2):
    """KL divergence between two dirichlet distributions"""
    d = spm_betaln(x2) - spm_betaln(x1) - np.sum((x2-x1)*spm_psi(x1+1/32),0)
    return np.sum(d)

def spm_betaln(X):
    """Generalized betaln function for unknown dimension matrixes
    The summation is on the first dimension so input matrix should be of 
    size A x B where B >= 1 and A >1 (ex : a 4 x 1 vector)"""
    X= np.squeeze(X)
    if (X.ndim == 1):
        X = X[X>0]
        return np.sum(gammaln(X))-gammaln(np.sum(X))
    else :
        shape = X.shape[1:]
        Y = np.zeros(shape)
        for i in (np.ndindex(shape)) :
            slicer = (slice(None),) + i
            Y[i] = spm_betaln(X[slicer])
        return Y

def spm_psi(X):
    return psi(X)-psi(np.sum(X,0))

def spm_dot(X,in2,i={}):
    """
    % Multidimensional dot (inner) product
    % FORMAT [Y] = spm_dot(X,x,[DIM])
    %
    % X   - numeric array
    % x   - np array or list
    % DIM - dimensions to omit (asumes ndims(X) = numel(x))
    %
    % Y  - inner product obtained by summing the products of X and x along DIM
    %
    % If DIM is not specified the leading dimensions of X are omitted.
    % If x is a vector the inner product is over the leading dimension of X
    %
    % See also: spm_cross
    %__________________________________________________________________________
    % Copyright (C) 2015 Wellcome Trust Centre for Neuroimaging
    
    % Karl Friston (brought to Python by C. ANNICCHIARICO)
    % $Id: spm_dot.m 7314 2018-05-19 10:13:25Z karl $
    """
    # initialise dimensions
    #--------------------------------------------------------------------------
    x = []
    if (type(in2)==list):
        # x is a list of vectors
        DIM = [y + X.ndim - len(in2) for y in range(len(in2))]     
        for k in range(len(in2)):
            x.append(np.copy(in2[k]))
    else :
        DIM = [0]
        x   = [np.copy(in2)]
    
    if not(i is spm_dot.__defaults__[0]):
        assert (i<=X.ndim), "Dimension to omit too large"
        assert (i<=len(x)), "The dimension to omit is beyond the vector shape"
        del DIM[i]
        del x[i]
    # inner product using recursive summation (and bsxfun)
    #--------------------------------------------------------------------------
    for d in range(len(x)) :
        s = np.ones((X.ndim)).astype(int)
        s[DIM[d]] = x[d].size
        x_reshaped = np.reshape(x[d],s)
        X = X*x_reshaped
        X = np.sum(X,axis=DIM[d],keepdims=True)
    #TODO : There is probably a better way to do this but no time for now
    # eliminate singleton dimensions
    #--------------------------------------------------------------------------
    X = np.squeeze(X)
    return X


def spm_kron(*argv) :
    n = (len(argv))
    ret = 0
    if (n>1):
        L = []
        for i in range(n):
            L.append(argv[i])
        return spm_kron(L)
    else :
        A  = argv[0]
        assert type(A)==list,"A must be list"
        assert len(A)>0,"A must have at least one element"
        ret = A[0]
        for i in range(1,len(A)) : 
            ret = np.kron(ret,A[i])
        return ret

def spm_dekron(X,Ns):
    # Roughly a way to "de-kronify" our state representation ?
    return spm_complete_margin(X.reshape(tuple(Ns)))


def spm_margin(X,factor):
    # marginalize joint distribution
    
    if (X.ndim == 1):
        assert (factor == 0), "The joint distribution shape " + str(X.shape) + " doesn't match the factor index " + str(factor)

        return X
    else :
        dimensions_to_sum_over = [i for i in range(X.ndim)if(i != factor)]
        return np.sum(X,tuple(dimensions_to_sum_over))

def spm_complete_margin(X):
    ret = []
    for factor in range(X.ndim):
        ret.append(spm_margin(X,factor))
    return ret 

def spm_all_combinations(*argv):
    matrix = combinations_m(*argv)
    out = []
    return matrix

def combinations_m(*argv):
    n = (len(argv)) 
    if(n==2):
        return combinations_two(argv[0],argv[1])[0]
    elif (n==1):
        return argv
    else :
        K = combinations_m(*argv[1:])
        return combinations_m(argv[0],K)

def combinations_two(A,B):
    combination_shape = A.shape + B.shape
    ret = np.zeros((combination_shape + (2,)))
    L = []
    for idx, x in np.ndenumerate(A):
        for idy, y in np.ndenumerate(B):
            ret[idx+idy,0] = x
            ret[idx+idy,1] = y
            L.append([x,y])
    return ret,L




def all_combinations(U,n):
    action_num = U.shape[0]
    xsize = action_num**n
    ysize = n
    output_matrix = np.zeros((xsize,ysize))
    for j in range(ysize):
        stepsize = action_num**(ysize-1-j)
        k = 0
        ind = 0
        print(stepsize)
        for i in range(xsize):
            print(str(k)+ "   " + str(i+1)  + " / "  + str(xsize) + "  " + str(j+1)+ " / "  + str(ysize))
            if (k<stepsize):
                output_matrix[i,j]=U[ind]
                k = k + 1
            else: 
                ind = ind + 1
                if(ind>=action_num):
                    ind = 0
                output_matrix[i,j]=U[ind]
                k = 1
    return output_matrix

def some_combinations(U,n,N):
    action_num = U.shape[0]
    xsize = N
    ysize = n
    output_matrix = np.zeros((xsize,ysize))
    for i in range(N):
        for j in range(n):
            k = random.randint(0,action_num-1)
            output_matrix[i,j] = U[k]
    return output_matrix