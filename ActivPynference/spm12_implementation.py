# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 13:44:10 2021

@author: Côme ANNICCHIARICO, adaptation of the work of :

%% Step by step introduction to building and using active inference models

% Supplementary Code for: A Step-by-Step Tutorial on Active Inference Modelling and its 
% Application to Empirical Data

% By: Ryan Smith, Karl J. Friston, Christopher J. Whyte
(MATLAB Script)
https://github.com/rssmith33/Active-Inference-Tutorial-Scripts/blob/main/Step_by_Step_AI_Guide.m
"""

import random as r
import time 
import numpy as np
import matplotlib.pyplot as plt
from function_toolbox import normalize,softmax,nat_log
from function_toolbox import spm_wnorm,cell_md_dot,md_dot, spm_cross,spm_KL_dir,spm_psi, spm_dot
from function_toolbox import G_epistemic_value

from miscellaneous_toolbox import isField
import sys 





class MDP_OPTIONS :
    def __init__(self):
        self.GAMMA = False

class MDP :
    
    # Agent constructor
    static_MDPcounter = 0
    def __init__(self,*args):
        
        if (len(args)<1) : #Initialize an agent
            self.seed = r.randrange(sys.maxsize)            
            self.id = self.static_MDPcounter
            self.static_MDPcounter = self.static_MDPcounter + 1
            self.name = 'default'
            self.savepath = ''
            self.options = MDP_OPTIONS()
            
            self.age = 0    #How many full iteration (experience + learning) has this MDP lived ?   
            
            self.packed = False
            self.ran = False
            
            # Simulation parameters
            self.T = 0
            self.Ni = 0
            
            
            #Model building blocks
            self.a_ = None       # A matrix beliefs (learning)
            self.b_ = None       # B matrix beliefs (learning)
            self.c_ = None       # C matrix beliefs (learning)
            self.d_ = None       # D matrix beliefs (learning)
            self.e_ = None       # E matrix beliefs (learning)
            
            self.A_ = None       # A matrix real (generative process)    
            self.B_ = None       # B matrix real (generative process)    
            self.C_ = None       # C matrix real (generative process)    
            self.D_ = None       # D matrix real (generative process)    
            self.E_ = None       # E matrix real (generative process)    
            
            self.V_ = None       # Allowable policies (T-1 x Np x F)
            self.U_ = None       # Allowable actions (1 x Np x F)
            
            #Optional outputs, might be overriden to force certain states
            self.s = None   # Sequence of true states
            self.o = None   # Sequence of observed outcomes
            self.O = None   # Likelihood of observed outcomes
            self.u = None   # Chosen Action
            self.n = None   # What is your utility ?
            
            #Modulation Parameters
            self.alpha = 512 # action precision
            self.beta = 1    # policy precision
            self.zeta = 3    # Occam window policies
            self.eta = 1     # learning rate
            self.tau = 4     # update time constant (gradient descent)
            self.chi = 1/64  # Occam window updates
            self.erp = 4    # update reset
            
            
            # Simularion Variables
            self.Np = 0    #Number of policies
            self.No = []   #Number of outcomes for each MODALITY
            self.Nmod = len(self.No) #Number of modalities
            self.Ns = []   #Number of states for each FACTOR
            self.Nf = len(self.Ns) #Number of factors
            self.Nu = 0     # Number of controllable transitions
            
            #Pure Output variables
            self.Fa = []            # Negative free energy of a
            self.Fb = []            # Negative free energy of b
            self.Fd = []            # Negative free energy of c
            self.Fe = []            # Negative free energy of d
            
            self.P = []             # Action posterior
            self.R = []             # Policy posterior
            
            self.Q = []             # State posteriors
            self.X = []             # BMA States            
            
            self.G = None     # EFE values
            self.F = None     # VFE values
            self.H = None     # Entropy value i guess ?
            
            self.vn = []            # Neuronal prediction error 
            self.xn = []            # Neuronal encoding of hidden states
            self.un = []            # Neuronal encoding of policies
            self.wn = None          # Neuronal encoding of policy precision
            self.dn = None          # Simulated dopamine response
            self.rt = None          # Simulated reaction times
        else : # This agent is the successor to a previous one
            parent_mdp = args[0]
            
            self.id = parent_mdp.id          # The same agent, but older
            self.age = parent_mdp.age + 1    #How many full iteration (experience + learning) has this MDP lived ?  
            self.seed = parent_mdp.seed
            

            try :
                i = int(parent_mdp.name[-1])
                if (type(i)==int) :
                    self.name = parent_mdp.name.split('_')[0] + '_gen' + str(self.age)
                    self.savepath = parent_mdp.savepath.split('_')[0] + '_gen' + str(self.age)
                else : 
                    self.name = parent_mdp.name + "_gen" + str(self.age)
                    self.savepath = parent_mdp.savepath + "_gen" + str(self.age)
            except :
                self.name = parent_mdp.name + "_gen" + str(self.age)
                self.savepath = parent_mdp.savepath + "_gen" + str(self.age)
            
            
            self.options = parent_mdp.options
            
            self.packed = False
            self.ran = False
            
            # Simulation parameters
            self.T = parent_mdp.T
            self.Ni =  parent_mdp.Ni
            
            def copy(parent):
                if (not(type(parent)==list) and not(type(parent)==np.ndarray)) :
                    if not(parent):
                        return None
                if(type(parent) == list):
                    returncopy = []
                    for i in range(len(parent)) :
                        returncopy.append(copy(parent[i]))
                    return returncopy
                else :
                    return np.copy(parent)
            
            #Model building blocks
            # TODO : check issues with array references copy
            self.a_ = copy(parent_mdp.a_)      # A matrix beliefs (learning)
            self.b_ = copy(parent_mdp.b_)       # B matrix beliefs (learning)
            self.c_ = copy(parent_mdp.c_)        # C matrix beliefs (learning)
            self.d_ = copy(parent_mdp.d_)        # D matrix beliefs (learning)
            self.e_ = copy(parent_mdp.e_)       # E matrix beliefs (learning)
            
            self.A_ = copy(parent_mdp.A_)       # A matrix real (generative process)    
            self.B_ = copy(parent_mdp.B_)       # B matrix real (generative process)    
            self.C_ = copy(parent_mdp.C_)       # C matrix real (generative process)    
            self.D_ = copy(parent_mdp.D_)       # D matrix real (generative process)    
            self.E_ = copy(parent_mdp.E_)       # E matrix real (generative process)    
            
            self.V_ = copy(parent_mdp.V_)       # Allowable policies (T-1 x Np x F)
            self.U_ = copy(parent_mdp.U_)       # Allowable actions (1 x Np x F)
            
            #Optional outputs, might be overriden to force certain states
            self.s = None   # Sequence of true states
            self.o = None   # Sequence of observed outcomes
            self.O = None   # Likelihood of observed outcomes
            self.u = None   # Chosen Action
            self.n = None   # What is your utility ?
            
            #Modulation Parameters
            self.alpha = parent_mdp.alpha # action precision
            self.beta = parent_mdp.beta    # policy precision
            self.zeta = parent_mdp.zeta    # Occam window policies
            self.eta = parent_mdp.eta     # learning rate
            self.tau = parent_mdp.tau     # update time constant (gradient descent)
            self.chi = parent_mdp.chi  # Occam window updates
            self.erp = parent_mdp.erp    # update reset
            
            
            # Simularion Variables
            self.Np = 0    #Number of policies
            self.No = []   #Number of outcomes for each MODALITY
            self.Nmod = len(self.No) #Number of modalities
            self.Ns = []   #Number of states for each FACTOR
            self.Nf = len(self.Ns) #Number of factors
            self.Nu = 0     # Number of controllable transitions
            
            #Pure Output variables (reinitialise all)
            self.Fa = []            # Negative free energy of a
            self.Fb = []            # Negative free energy of b
            self.Fd = []            # Negative free energy of c
            self.Fe = []            # Negative free energy of d
            
            self.P = []             # Action posterior
            self.R = []             # Policy posterior
            
            self.Q = []             # State posteriors
            self.X = []             # BMA States            
            
            self.G = None     # EFE values
            self.F = None     # VFE values
            self.H = None     # Entropy value i guess ?
            
            self.vn = []            # Neuronal prediction error 
            self.xn = []            # Neuronal encoding of hidden states
            self.un = []            # Neuronal encoding of policies
            self.wn = None          # Neuronal encoding of policy precision
            self.dn = None          # Simulated dopamine response
            self.rt = None          # Simulated reaction times
    
    def copy(self):
        returnmdp = MDP()
        parent_mdp = self
    
        returnmdp.id = self.static_MDPcounter
        self.static_MDPcounter = self.static_MDPcounter + 1
        
        returnmdp.age = parent_mdp.age    #How many full iteration (experience + learning) has this MDP lived ?  
        returnmdp.seed = parent_mdp.seed
        

        returnmdp.name = self.name + "_copy"
        returnmdp.savepath = self.savepath + "_copy"
        
        returnmdp.packed = False
        self.ran = returnmdp.ran
        
        # Simulation parameters
        returnmdp.T = parent_mdp.T
        returnmdp.Ni =  parent_mdp.Ni
        
        def copy(parent):
            if (not(type(parent)==list) and not(type(parent)==np.ndarray)) :
                if not(parent):
                    return None
            if(type(parent) == list):
                returncopy = []
                for i in range(len(parent)) :
                    returncopy.append(copy(parent[i]))
                return returncopy
            else :
                return np.copy(parent)
        #Model building blocks
        # TODO : check issues with array references copy
        returnmdp.a_ = copy(parent_mdp.a_)      # A matrix beliefs (learning)
        returnmdp.b_ = copy(parent_mdp.b_)       # B matrix beliefs (learning)
        returnmdp.c_ = copy(parent_mdp.c_)        # C matrix beliefs (learning)
        returnmdp.d_ = copy(parent_mdp.d_)        # D matrix beliefs (learning)
        returnmdp.e_ = copy(parent_mdp.e_)       # E matrix beliefs (learning)
        
        returnmdp.A_ = copy(parent_mdp.A_)       # A matrix real (generative process)    
        returnmdp.B_ = copy(parent_mdp.B_)       # B matrix real (generative process)    
        returnmdp.C_ = copy(parent_mdp.C_)       # C matrix real (generative process)    
        returnmdp.D_ = copy(parent_mdp.D_)       # D matrix real (generative process)    
        returnmdp.E_ = copy(parent_mdp.E_)       # E matrix real (generative process)    
        
        returnmdp.V_ = copy(parent_mdp.V_)       # Allowable policies (T-1 x Np x F)
        returnmdp.U_ = copy(parent_mdp.U_)       # Allowable actions (1 x Np x F)
        
        #Optional outputs, might be overriden to force certain states
        returnmdp.s = None   # Sequence of true states
        returnmdp.o = None   # Sequence of observed outcomes
        returnmdp.O = None   # Likelihood of observed outcomes
        returnmdp.u = None   # Chosen Action
        returnmdp.n = None   # What is your utility ?
        
        #Modulation Parameters
        returnmdp.alpha = parent_mdp.alpha # action precision
        returnmdp.beta = parent_mdp.beta    # policy precision
        returnmdp.zeta = parent_mdp.zeta    # Occam window policies
        returnmdp.eta = parent_mdp.eta     # learning rate
        returnmdp.tau = parent_mdp.tau     # update time constant (gradient descent)
        returnmdp.chi = parent_mdp.chi  # Occam window updates
        returnmdp.erp = parent_mdp.erp    # update reset
        
        
        # Simularion Variables
        returnmdp.Np = 0    #Number of policies
        returnmdp.No = []   #Number of outcomes for each MODALITY
        returnmdp.Nmod = len(self.No) #Number of modalities
        returnmdp.Ns = []   #Number of states for each FACTOR
        returnmdp.Nf = len(self.Ns) #Number of factors
        returnmdp.Nu = 0     # Number of controllable transitions
        
        #Pure Output variables (reinitialise all)
        returnmdp.Fa = []            # Negative free energy of a
        returnmdp.Fb = []            # Negative free energy of b
        returnmdp.Fd = []            # Negative free energy of c
        returnmdp.Fe = []            # Negative free energy of d
        
        returnmdp.P = []             # Action posterior
        returnmdp.R = []             # Policy posterior
        
        returnmdp.Q = []             # State posteriors
        returnmdp.X = []             # BMA States            
        
        returnmdp.G = None     # EFE values
        returnmdp.F = None     # VFE values
        returnmdp.H = None     # Entropy value i guess ?
        
        returnmdp.vn = []            # Neuronal prediction error 
        returnmdp.xn = []            # Neuronal encoding of hidden states
        returnmdp.un = []            # Neuronal encoding of policies
        returnmdp.wn = None          # Neuronal encoding of policy precision
        returnmdp.dn = None          # Simulated dopamine response
        returnmdp.rt = None          # Simulated reaction times

        return returnmdp
    
    def pack_model(self):
        """This function checks building blocks and initializes all basic variables needed for the experience. It is called before proceeding to the run"""
        checker = True
        checker = checker and self.A_ and self.B_ and self.C_ and self.D_
        assert checker,"The model has not been built properly (basic building blocks). Aborting model packing."
        checker = checker and (self.T > 0)
        assert checker,"The total simulation time cannot be <1. Aborting model packing."
        checker = checker and (self.Ni > 0)
        assert checker,"The iteration number cannot be <1. Aborting model packing."
        
        self.Np = self.V_.shape[1]
        
        self.Nmod = len(self.A_)
        self.No = []
        for i in range(self.Nmod) :
            self.No.append(self.A_[i].shape[0])
            
        self.Nf = len(self.D_)
        self.Ns = []
        for i in range(self.Nf):
            self.Ns.append(self.D_[i].shape[0])
        
        self.Nu = []  
        for f in range(self.Nf) :
            self.Nu.append(self.B_[f].shape[2])
            # B_[f] shoulc always be a 3D matrix
        
        if(self.name == '') :
            self.name = 'unnamed_model'
    
    def initialize_model_vars(self):
        #TODO : Initialize output variables for a run
        
        self.F = np.zeros((self.Np,self.T))
        self.G = np.zeros((self.Np,self.T))
        self.H = np.zeros((self.T,))
        
        self.Fa = []            # Negative free energy of a
        self.Fb = []            # Negative free energy of b
        self.Fd = []            # Negative free energy of c
        self.Fe = []            # Negative free energy of d
        
        self.P = []             # Action posterior
        self.R = []             # Policy posterior
        
        self.Q = []             # State posteriors
        self.X = []             # BMA States            
        
        self.vn = []            # Neuronal prediction error 
        self.xn = []            # Neuronal encoding of hidden states
        self.un = []            # Neuronal encoding of policies
        self.wn = None          # Neuronal encoding of policy precision
        self.dn = None          # Simulated dopamine response
        self.rt = None          # Simulated reaction times
        
        
        
        
        self.packed = True
        print("Model packed.")
        
    
    def run(self,verbose=0):
        print("----------------------------------------------")
        print("        MODEL " + str(self.name) )
        print()
        print("seed: " + str(self.seed))
        print("----------------------------------------------")
        
        
        
        
        if (not(self.packed)) :
            self.pack_model()
        self.initialize_model_vars()
        #Simulation variables
        T = self.T
        Ni = self.Ni
        
        Nf = self.Nf
        Ns = self.Ns
        
        Nmod = self.Nmod
        No = self.No
        
        Np = self.Np
        Nu = self.Nu
        
        #TimeConst = self.time_constant # Replaced by self.tau
    
        # Likelihood model a / A
        if isField(self.a_):
            a = normalize(self.a_)
            
            a_prior = []
            a_complexity = []
            for modality in range(Nmod):
                a_prior.append(np.copy(self.a_[modality]))
                a_complexity.append( spm_wnorm(a_prior[modality])*(a_prior[modality]>0) )
        else :
            a = normalize(self.A_)
        A = normalize(self.A_)
        
        
        # Transition model b / B
        if isField(self.b_): # If we are learning the transition matrices
            b = normalize(self.b_)
            
            b_prior = []
            b_complexity = []
            for factor in range(Nf):   # For all factors
                b_prior.append(np.copy(self.b_[factor]))
                b_complexity.append(spm_wnorm(b_prior[factor])*(b_prior[factor]>0))
        else :
            b = normalize(self.B_)
        B = normalize(self.B_)
        
        # prior over initial states d/D
        if isField(self.d_):
            d = normalize(self.d_)
            
            d_prior = []
            d_complexity = []
            for f in range(Nf):
                d_prior.append(np.copy(self.d_[f]))
                d_complexity.append(spm_wnorm(d_prior[f]))
        elif isField(self.D_) :
            d = normalize(self.D_)
        else :
            d = []
            for f in range(Nf):
                d.append(normalize(np.ones(Ns[f],)))
            self.D_ = d
        D = normalize(self.D_)
        
        # Habit E
        if isField(self.e_):
            E = self.e_
            e_prior = np.copy(self.e_)
        elif isField(self.E_) :
            E = self.E_
        else :
            E = np.ones((Np,))
        E = E/sum(E)
        
        # Preferences C
        C = []
        if isField(self.c_):
            c_prior = []
            for modality in range(Nmod):
                C.append(spm_psi(self.c_[modality] + 1./32))
                c_prior.append(np.copy(self.c_[modality]))      
        elif isField(self.C_):
            for modality in range(Nmod):
                C.append(self.C_[modality])
        else : 
            for modality in range(Nmod):
                C.append(np.zeros((No[modality],1)))
            
        for modality in range(Nmod):
            if (C[modality].shape[1] == 1) :
                C[modality] = np.tile(C[modality],(1,T))
                if (self.c_):
                    self.c_[modality] = np.tile(self.c_[modality],(1,T))
                    c_prior[modality] = np.tile(c_prior[modality],(1,T))
            
            C[modality] = nat_log(softmax(C[modality],0))
        
        
        V = self.V_
        
         
        #OUTCOMES
        O = []
        for modality in range(Nmod):
            O.append(np.zeros((No[modality],T)))
                                                    
        o = np.full((Nmod,T),-1)
                                    # the value inside true states for a given factor can be :
                                    # <0 if not yet defined : it will be created by the generative process
                                    # x in [0,No[modality]-1] for a given modality at a given time if we want to explicit some value
        if isField(self.o):  # There is an "o" matrix with fixed values ?
            o[self.o >=0] = self.o[self.o>=0]
            # If there are fixed values for the observations, they will be copied
        self.o = np.copy(o)
        
        #states
        s = np.full((Nf,T),-1)  # = true states
        if isField(self.s):
            s[self.s >=0] = self.s[self.s >=0]
        self.s = np.copy(s)
        #s = s.astype(np.int)  
        
        u_temp = np.full((Nf,T-1),-1)  #chosen action
        if isField(self.u):
            u_temp[self.u>=0] = self.u[self.u>=0]
        self.u = u_temp
        
        # Posterior expectations of hidden states
        xn = []
        vn = []
        x = []
        X = []
        for f in range(Nf):        
            xn.append(np.zeros((Ni,Ns[f],T,T,Np)) + 1./Ns[f])
            vn.append(np.zeros((Ni,Ns[f],T,T,Np)))
            x.append(np.zeros((Ns[f],T,Np)) + 1./Ns[f])   # Posterior expectation of all hidden states
            X.append(np.tile(np.reshape(d[f],(-1,1)),(1,T)))
            #X.append(np.expand_dims(d[f],1))
            for k in range(Np):
                x[f][:,0,k] = d[f]
           
                
        
        #Posterior over policies and action
        P = np.zeros(tuple(Nu)+(T-1,))
        un = np.zeros((Np,Ni*T))
        u = np.zeros((Np,T))                 # = posterior over action
        if (Np == 1) :
            u = np.ones((Np,T))
        # -------------------------------------------------------------
        p = np.zeros((Np,))
        for policy in range(Np): # Indices of allowable policies
            p[policy] = policy
        p = p.astype(np.int)
            
            
        L = []
        
        # expected rate parameter (precision of posterior over policies)
        #----------------------------------------------------------------------
        qb = self.beta #                      % initialise rate parameters
        #w  = 1/qb   #                   % posterior precision (policy)
        w = np.zeros((T,))
        w[0] = 1/qb
        
        wn = np.zeros((Ni*T,))
        
        reaction_time = np.zeros((T,))

        for t in range(T) :
            announcer = "\r     Experience in progress (t = " + str(t+1) + " / " + str(T) + " )."
            print(announcer,end='')
            
        #    print()
        #    print()
            
            #GENERATIVE PROCESS !!
            #-------------------------------------------------------------------------------------------------
            # generate hidden states and outcomes
            #======================================================================
            # Note for later : 
            # A is the ground truth for observation generation process
            # a is an approximation learnt by our agent in the generative model (if the option is enabled)
            # Same for letters B,(D ?)
            
            
            for f in range(Nf) :
            #        Here we sample from the prior distribution over states to obtain the
            #        % state at each time point. At T = 1 we sample from the D vector, and at
            #        % time T > 1 we sample from the B matrix. To do this we make a vector 
            #        % containing the cumulative sum of the columns (which we know sum to one), 
            #        % generate a random number (0-1),and then use the find function to take 
            #        % the first number in the cumulative sum vector that is >= the random number. 
            #        % For example if our D vector is [.5 .5] 50% of the time the element of the 
            #        % vector corresponding to the state one will be >= to the random number. 
                if (self.s[f,t] < 0) :
                    if (t==0) :
                        prob_state = D[f]
                    else :
                        prob_state = B[f][:,self.s[f,t-1],self.u[f,t-1]]
                    self.s[f,t] = np.argwhere(r.random() <= np.cumsum(prob_state,axis=0))[0]
            self.s[:,t] = self.s[:,t].astype(np.int)
            
            #Posterior predictive density over hidden external states
            xq = []
            xqq = []
            for f in range(Nf) :
                if (t ==0) :
                    xqq.append(X[f][:,t])
                else :
                    xqq.append(np.dot(np.squeeze(b[f][:,:,self.u[f,t-1]]),X[f][:,t-1]))
                xq.append(X[f][:,t])
            
            
            #observations
            for modality in range(Nmod):
                if (self.o[modality,t] < 0):
                    ind = (slice(None),) + tuple(self.s[:,t])  # Indice corresponding to the current active states
                    self.o[modality,t] = np.argwhere(r.random() <= np.cumsum(A[modality][ind],axis=0))[0]
            self.o[:,t] = self.o[:,t].astype(np.int)
            
            # Get probabilistic outcomes from samples
            for modality in range(Nmod):
                vec = np.zeros((1,No[modality]))
                #print(vec.shape)
                index = self.o[modality,t]
                #print(index)
                vec[0,index] = 1
                O[modality][:,t] = vec
                
           
                
            # Likelihood of hidden states
            L.append(1)
            for modality in range (Nmod):
                L[t] = L[t] * spm_dot(a[modality],O[modality][:,t])
            # END OF GENERATIVE PROCESS !!
            #-------------------------------------------------------------------------------------------------
                
            if (isField(self.zeta)):
                if not(isField(self.U_)) and (t>0):
                    F = nat_log(u[p,t-1])
                    p = p[(F-np.max(F))>-self.zeta]                
            tstart = time.time()
            for f in range(Nf):
                x[f] = softmax(nat_log(x[f])/self.erp,axis = 0,center = False)
                #print(softmax(nat_log(x[f])/self.erp,axis = 0,center = True)[:,:,0])

            
            
            #GENERATIVE MODEL !!
            #------------------------------------------------------------------------------------------------
            
            
            
            # % Variational updates (hidden states) under sequential policies
            #%==============================================================
            S = V.shape[0] + 1
            
            if (self.U_):
                R = t;
            else :
                R = S
            F = np.zeros((Np,))
            G = np.zeros((Np,))
            # marginal message passing (minimize F and infer posterior over states)
            #----------------------------------------------------------------------
            
            
            
            
                #print(nat_log(np.dot(np.squeeze(b[0][:,:,V[1-1,0,0]]),x[0][:,1-1,0])))
                
            
            for policy in p :
                dF = 1 # Criterion for given policy
                for iteration in range(Ni) :
                    
                    F[policy] = 0
                    
                    for tau in range(S): #Loop over future time points
                        
                        #posterior over outcomes
                        if (tau <= t) :
                            for factor in range(Nf):
                                xq[factor]=np.copy(x[factor][:,tau,policy])
                        for factor in range (Nf):
                            #  hidden state for this time and policy
                            sx = np.copy(x[factor][:,tau,policy])
                            qL = np.zeros((Ns[factor],))
                            v = np.zeros((Ns[factor],))
                            
                            # evaluate free energy and gradients (v = dFdx)
                            if ((dF > np.exp(-8)) or (iteration > 3)) :
                                
                                # marginal likelihood over outcome factors
                                if (tau <= t) :
                                    qL = spm_dot(L[tau],xq,factor)
                                    qL = nat_log(qL)
                                qx = nat_log(sx)
                                
                                
                                #Empirical priors (forward messages)
                                if (tau == 0):
                                    px = nat_log(d[factor])
                                else :
                                    px = nat_log(np.dot(np.squeeze(b[factor][:,:,V[tau-1,policy,factor]]),x[factor][:,tau-1,policy]))
                                v = v +px + qL - qx
                                
                                #Empirical priors (backward messages)
                                if (tau == R-1) :
                                    px = 0
                                else : 
                                    px = nat_log(np.dot(normalize(b[factor][:,:,V[tau,policy,factor]].T),x[factor][:,tau+1,policy]))
                                    v = v +px + qL - qx
                                
                                if ((tau==0) or (tau==S-1)):
                                    F[policy] = F[policy] + 0.5*np.dot(sx.T,v)
                                else :
                                    F[policy] = F[policy] + np.dot(sx.T,0.5*v - (Nf-1)*qL/Nf)
                                    
                                v = v - np.mean(v)
                                sx = softmax(qx + v/self.tau)

                            else :
                                F[policy] = G[policy] # End of condition
                                
                            x[factor][:,tau,policy] = sx
                            xq[factor] = np.copy(sx)
                            xn[factor][iteration,:,tau,t,policy] = sx
                            vn[factor][iteration,:,tau,t,policy] = v
                            
                    # end of loop onn tau --> convergence :
                    if (iteration > 0):
                        dF = F[policy] - G[policy]
                    G = np.copy(F)
                # End of loop on iterations
            #End of loop on policies    
            
#            if (t==1):
#                print(x[0][:,:,0])
#                print(x[0][:,:,1])
#                print(x[0][:,:,2])
#                print(x[0][:,:,3])
#                print(x[0][:,:,4])
#            print("G : ")
#            print(G)
#            print()
#            print()
            # EFE : 
            pu = 1 #Empirical prior
            qu = 1 #posterior
            Q = np.zeros((Np,)) # Actual EFE
            
            if (Np>1) :
                for policy in p:
                    # Bayesian surprise about initial conditions
                    if isField(self.d_):
                        for factor in range (Nf):
                            Q[policy] = Q[policy] - spm_dot(d_complexity[factor],x[factor][:,0,policy])
                    for timestep in range(t,S):
                        for factor in range (Nf):
                            xq[factor] = x[factor][:,timestep,policy]
                        
                        
                        #Bayesian surprise about states
                        Q[policy] = Q[policy] + G_epistemic_value(a,xq) 
                        for modality in range(Nmod):
                            
                            #Prior preferences about outcomes
                            qo = spm_dot(a[modality],xq)   #predictive observation posterior
                            Q[policy] = Q[policy] + np.dot(qo.T,C[modality][:,timestep])
                            #Bayesian surprise about parameters
                            if isField(self.a_):
                                Q[policy] = Q[policy] - spm_dot(a_complexity[modality],[qo]  + xq[:])
#                                                                 [predictive_observation_posterior] + Expected_states[:])
                        #print(Q[policy])
                        #End of loop on policies                                
                            
                if (t>0):
                    w[t] = w[t-1]
                
                for iteration in range(Ni):
                    # posterior and prior beliefs about policies
                    qu = softmax(nat_log(E)[p] + w[t]*Q[p] + F[p])
                    pu = softmax(nat_log(E)[p] + w[t]*Q[p])
                    if (self.options.GAMMA):
                        w[t] = 1/self.beta
                    else :
                        eg = np.dot((qu-pu).T,Q[p])
                        dFdg = qb - self.beta + eg
                        qb = qb - dFdg/2.
                        w[t] = 1/qb
                    
                    #dopamine responses
                    n = t*Ni + iteration
                    wn[n] = w[t]
                    un[p,n] = qu
                    u[p,t] = qu      # Policy posterior
            
            # BMA over hidden states
            for factor in range(Nf):
                for tau in range(S):
                    X[factor][:,tau] =np.dot(x[factor][:,tau,:],u[:,t])
            
            reaction_time[t] = time.time() - tstart
            
            self.F[:,t] = F
            self.G[:,t] = Q
            self.H[t] = np.dot(qu.T,self.F[p,t]) - np.dot(qu.T,(nat_log(qu) - nat_log(pu))) ;
                
            # TODO : check for residual uncertainty (in hierarchical schemes) + VOX mode
#            if isfield(MDP,'factor')
#                
#                for f = MDP(m).factor(:)'
#                    qx     = X{m,f}(:,1);
#                    H(m,f) = qx'*spm_log(qx);
#                end
#                
#                % break if there is no further uncertainty to resolve
#                %----------------------------------------------------------
#                if sum(H(:)) > - chi && ~isfield(MDP,'VOX')
#                    T = t;
#                end
#            end
            
            # Action selection !!
            if (t<T-1):
                #Marginal posterior over action
                Pu = np.zeros(tuple(Nu))  # Action posterior intermediate
                
                for policy in range(Np):
                    
                    sub = V[t,policy,:] # coordinates for the corresponding action wrt t and policy
                    action_coordinate = tuple(sub)
                    Pu[action_coordinate] = Pu[action_coordinate] + u[policy,t]
                
                Pu = softmax(self.alpha*nat_log(Pu))
                P[...,t] = Pu
                #action_posterior[...,t] = action_posterior_intermediate
                # Next action : sampled from marginal posterior
                for factor in range(Nf):
                    if (self.u[factor,t]<0) : # The choice of action is not overriden
                        if(Nu[factor]>1) :
                            randomfloat = r.random()
                            #print(np.cumsum(action_posterior_intermediate,axis=factor))
                            ind = np.argwhere(randomfloat <= np.cumsum(Pu,axis=factor))[0]
                            #ind est de taille n où n est le nombre de transitions à faire :
                            self.u[factor,t] = ind[factor]
                            #print(b, np.cumsum(action_posterior_intermediate,axis=1),ind[0])                
                        else :
                            self.u[factor,t] = 0
                if isField(self.U_) :
                    for factor in range(Nf):
                        V[t,:,factor] = self.u[factor,t]
                    
                    for j in range (self.U_.shape[0]) :
                        if (t+1 < T-1) :
                            V[t+1,:,:] = self.U_[:,:]
                        
                    for factor in range(Nf):
                        for policy in range (Np):
                            x[factor][:,:,policy] = 1.0/Ns[factor]
                #End of condition on U_
            # End of Action selection
#            print(self.F)
#            print(self.G)
#            print(self.u)
#            print()
            if (t==T-1): #Accumulate all evidences
                if (T==1):
                    self.u = np.zeros((Nf,1)) 
                self.o = self.o[:,0:T]
                self.s = self.s[:,0:T]
                self.u = self.u[:,0:T-1]
            
        #End of loop over time
        
        print("\nExperience ended without errors.")

        # LEARNING :
        for t in range(T):
            if isField(self.a_): 
                for modality in range(Nmod):
                    da = (O[modality][:,t])
                    for factor in range(Nf):
                        da = spm_cross(da,X[factor][:,t])
                    da = da*(self.a_[modality]>0)        
                    self.a_[modality] = self.a_[modality] + da*self.eta
                    
            if isField(self.b_)and (t>0) :
                for factor in range(Nf):
                    for policy in range (Np):
                        v = V[t-1,policy,factor]
                        db = u[policy,t]*np.outer(x[factor][:,t,policy],x[factor][:,t-1,policy].T)
                        db = db*(self.b_[factor][:,:,v]>0)
                        self.b_[factor][:,:,v] = self.b_[factor][:,:,v] + db*self.eta 
                        
            if isField(self.c_) :
                for modality in range(Nmod):
                    dc = O[modality][:,t]
                    if (self.c_[modality].shape[1]>1) : #If preferences are dependent on time
                        dc = dc*(self.c_[modality][:,t]>0)
                        self.c_[modality][:,t] = self.c_[modality][:,t] + dc*self.eta
                    else : 
                        dc = dc*(self.c_[modality] > 0)
                        self.c_[modality] = self.c_[modality] + dc* self.eta
        
        if isField(self.d_) : #Update initial hidden states beliefs
            for factor in range (Nf):
                i = self.d_[factor]>0
                self.d_[factor][i] = self.d_[factor][i] + X[factor][i,0]
            
            
        if isField(self.e_) : # Update agent habits
            self.e_ = self.e_ + u[:,T-1]*self.eta;
        #♣print(u)
        
        # Negative freeee eneergiiiies
        for modality in range (Nmod):
            if isField(self.a_):
                self.Fa.append(-spm_KL_dir(self.a_[modality],a_prior[modality]))
            if isField(self.c_) :
                self.Fc.append(- spm_KL_dir(self.c_[modality],c_prior[modality]))
        
        for factor in range(Nf):
            if isField(self.b_):
                self.Fb.append(-spm_KL_dir(self.b_[factor],b_prior[factor]))
            if isField(self.d_):
                self.Fd.append(-spm_KL_dir(self.d_[factor],d_prior[factor]))
        
        if (Np>1):
            dn = 8*np.gradient(wn) + wn/8.0
        else :
            dn = None
            wn = None
        
        Xn = []
        Vn = []
        # BMA Hidden states
        for factor in range(Nf):
            Xn.append(np.zeros((Ni,Ns[factor],T,T)))
            Vn.append(np.zeros((Ni,Ns[factor],T,T)))
            
            for t in range(T):
                for policy in range(Np):
                    Xn[factor][:,:,:,t] = Xn[factor][:,:,:,t] + np.dot(xn[factor][:,:,:,t,policy],u[policy,t])
                    Vn[factor][:,:,:,t] = Vn[factor][:,:,:,t] + np.dot(vn[factor][:,:,:,t,policy],u[policy,t])
        print("Learning and encoding ended without errors.")
        
        if isField(self.U_):
            u = u[:,:-1]
            un =  un[:,:-Ni]

        self.O = O  #Outcomes
        self.P = P #Probability of action at time t
        self.R = u #Conditionnal expectation over policies (posterior over policies)
        
        self.Q = x #posterior over states
        self.X = X # BMA states
        
        self.C_ = C
        
        self.w = w #Posterior over precision
        self.wn = wn #neuronal encoding of precision
        
        self.vn = Vn #neuronal prediction error
        self.xn = xn #neuronal encoding of hidden states
        self.un = un #neuronal encoding of policies
        self.dn = dn # dopamine responses (deconvolved)
        self.rt = reaction_time
#        print(self.u)
#        plt.plot(np.linspace(0,dn.shape[0],dn.shape[0]),dn)
#        plt.show()
        self.ran = True


    def run_for_N_generations(self,N):
        r.seed(self.seed)
        mdp = self
        mdp_list = []
        for i in range(N):
            mdp_list.append(mdp)
            mdp.run()
            mdp = MDP(mdp)
        return mdp_list




