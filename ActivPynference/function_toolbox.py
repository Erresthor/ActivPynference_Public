# -*- coding: utf-8 -*-
"""
Created on Fri May  7 11:58:11 2021

@author: cjsan
"""
import numpy as np
from scipy.special import gammaln
from scipy.special import psi


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

def normalize(X,axis=0,epsilon=1e-10):                                                               ###normalises a matrix of probabilities
    if(type(X) == list):
        x =[]
        for i in range(len(X)):
            x.append(normalize(X[i],axis=axis))
    else :
        X = X.astype(float)
        Xint =  np.sum(X,axis,keepdims=True) < epsilon
        Xint = np.repeat(Xint,X.shape[axis],axis=axis)
        X[Xint] = X[Xint] + epsilon
        x= X/(np.sum(X,axis))
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


#
#
#
#
#T = 3 
#T = T
#Ni = 16
#
## Priors about initial states
## Prior probabilities about initial states in the generative process
#D_ =[]
## Context state factor
#D_.append(np.array([1,0])) #[Left better, right better]
## Behaviour state factor
#D_.append(np.array([1,0,0,0])) #{'start','hint','choose-left','choose-right'}
#D_ = D_
#
## Prior beliefs about initial states in the generative process
#d_ =[]
## Context beliefs
#d_.append(np.array([0.25,0.25])) #[Left better, right better]
## Behaviour beliefs
#d_.append(np.array([1,0,0,0])) #{'start','hint','choose-left','choose-right'}
#d_ = d_
#
## State Outcome mapping and beliefs
## Prior probabilities about initial states in the generative process
#Ns = [D_[0].shape[0],D_[1].shape[0]] #(Number of states)
#A_ = []
##Mapping from states to observed hints, accross behaviour states (non represented)
##
## [ .  . ]  No hint
## [ .  . ]  Machine Left Hint            Rows = observations
## [ .  . ]  Machine Right Hint
## Left Right
## Columns = context state
#A_obs_hints = np.zeros((3,Ns[0],Ns[1]))
#A_obs_hints[0,:,:] = 1
#pHA = 1
#A_obs_hints[:,:,1] = np.array([[0,0],
#                         [pHA, 1-pHA],
#                         [1-pHA,pHA]]) # Behaviour ste "hint" gives an observed hint
#    
#    
##Mapping from states to outcome (win / loss / null), accross behaviour states (non represented)
##
## [ .  . ]  Null
## [ .  . ]  Win           Rows = observations
## [ .  . ]  Loss
##
## Columns = context state
#A_obs_outcome = np.zeros((3,Ns[0],Ns[1]))
#A_obs_outcome[0,:,0:2] = 1
#pWin = 1
#A_obs_outcome[:,:,2] = np.array([[0,0],   # If we choose left, what is the probability of achieving win / loss 
#                         [pWin, 1-pWin],
#                         [1-pWin,pWin]]) # Choice gives an observable outcome
#               # If true = left, right
#A_obs_outcome[:,:,3] = np.array([[0,0],     # If we choose right, what is the probability of achieving win / loss 
#                         [1-pWin, pWin],
#                         [pWin,1-pWin]]) # Choice gives an observable outcome
#              # If true = left, right
#
##Mapping from behaviour states to observed behaviour
##
## [ .  .  .  .] start
## [ .  .  .  .] hint
## [ .  .  .  .] choose left         Row = Behaviour state
## [ .  .  .  .] choose right
##  s   h  l  r
##
## 3rd dimension = observed behaviour
## The 2nd dimension maps the dependance on context state
#A_obs_behaviour = np.zeros((Ns[1],Ns[0],Ns[1]))
#for i in range (Ns[1]) :
#    A_obs_behaviour[i,:,i] = np.array([1,1])
#
#
#A_ = [A_obs_hints,A_obs_outcome,A_obs_behaviour]
#A_ = A_
#
## Transition matrixes between hidden states ( = control states)
#B_ = []
##a. Transition between context states --> The agent cannot act so there is only one :
#B_context_states = np.array([[[1],[0]],
#                             [[0],[1]]])
#B_.append(B_context_states)
##b. Transition between behavioural states --> 4 actions
#B_behav_states = np.zeros((Ns[1],Ns[1],Ns[1]))
## - 0 --> Move to start from any state
#B_behav_states[0,:,0] = 1
## - 1 --> Move to hint from any state
#B_behav_states[1,:,1] = 1
## - 2 --> Move to choose left from any state
#B_behav_states[2,:,2] = 1
## - 3 --> Move to choose right from any state
#B_behav_states[3,:,3] = 1
#B_.append(B_behav_states)
#B_  = B_
#
## Preferred outcomes
## One matrix per outcome modality. Each row is an observation, and each
## columns is a time point. Negative values indicate lower preference,
## positive values indicate a high preference. Stronger preferences promote
## risky choices and reduced information-seeking.
#No = [A_[0].shape[0],A_[1].shape[0],A_[2].shape[0]]
#
#
#
#la = 1 #Loss aversion
#rs = 3 #reward seeking
#
#C_hints = np.zeros((No[0],T))
#C_win_loss = np.zeros((No[1],T))
#C_win_loss = np.array([[0,0,0],     #null
#                       [0,rs,rs/2],  #win
#                       [0,-la,-la]]) #loss
#C_observed_behaviour = np.zeros((No[2],T))
#C_ = [C_hints,C_win_loss,C_observed_behaviour]
#C_ = C_
#
#
## Policies
#print(" Allowable policies : U / V")
#Np = 5 #Number of policies
#Nf = 2 #Number of state factors
#V_ = np.zeros((T-1,Np,Nf))
#V_[:,:,0]= np.array([[0,0,0,0,0],      # T = 2
#                     [0,0,0,0,0]])     # T = 3  row = time point
#    #                colums = possible course of action in this modality (0 -->context states)
#V_[:,:,1] = np.array([[0,1,1,2,3],      # T = 2
#                     [0,2,3,0,0]])     # T = 3  row = time point in this modality (1 -->behavioural states)
#    #                colums = possible course of action
#V_ = V_.astype(np.int)
#V_ = V_
#
##Habits
#E_ = None
#E_ = E_
#
#
#eta = 1 #Learning rate
#beta = 1 # expected precision in EFE(pi), higher beta --> lower expected precision
#         # low beta --> high influence of habits, less deterministic policiy selection
#alpha = 32 # Inverse temperature / Action precision
#            # How much randomness in selecting actions
#            # (high alpha --> more deterministic)
#erp = 4  # degree of belief resetter at each time point
#         # ( 1 means no reset bet. time points, higher values mean more loss in confidence)
#tau = 4 #Time constant for evidence accumulation
#            # magnitude of updates at each iteration of gradient descent
#            # high tau --> smaller updates, smaller convergence, greater  stability
#
#
#
#a_ = []
#for mod in range (len(A_)):
#    a_.append(np.copy(A_[mod])*200)
#
#a_[0][:,:,1] = np.array([[0,0],
#                        [0.25,0],
#                        [0.25,0]])
#
#K = np.array([[1,2,3,0],
#              [4,5,6,0]])
#
#a = normalize(K,0)
#
#print(a)




