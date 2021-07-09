# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 13:44:10 2021

@author: Côme ANNICCHIARICO(come.annicchiarico@mines-paristech.fr), adaptation of the work of :

%% Step by step introduction to building and using active inference models

% Supplementary Code for: A Step-by-Step Tutorial on Active Inference Modelling and its 
% Application to Empirical Data

% By: Ryan Smith, Karl J. Friston, Christopher J. Whyte
(MATLAB Script)
https://github.com/rssmith33/Active-Inference-Tutorial-Scripts/blob/main/Step_by_Step_AI_Guide.m


AND 

Towards a computational (neuro)phenomenology of mental action: modelling
meta-awareness and attentional control with deep-parametric active inference (2021)

Lars Sandved-Smith  (lars.sandvedsmith@gmail.com)
Casper Hesp  (c.hesp@uva.nl)
Jérémie Mattout  (jeremie.mattout@inserm.fr)
Karl Friston (k.friston@ucl.ac.uk)
Antoine Lutz (antoine.lutz@inserm.fr)
Maxwell J. D. Ramstead (maxwell.ramstead@mcgill.ca)


------------------------------------------------------------------------------------------------------
Generic class to build a multilayered model. This layer uses upper level layers outcomes to compute precisions used to weight blocks during inference.

Long term objective --> A comprehensive class able to simulate both "emotional/attentional" cognitive levels AND contextual levels.


"""
import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import random as r
import time 
import numpy as np
import matplotlib.pyplot as plt
from function_toolbox import normalize,softmax,nat_log,precision_weight
from function_toolbox import spm_wnorm,cell_md_dot,md_dot, spm_cross,spm_KL_dir,spm_psi, spm_dot
from function_toolbox import G_epistemic_value
from plotting_toolbox import basic_autoplot
from miscellaneous_toolbox import isField
from explore_exploit_model import explore_exploit_model

class layer_parameters :
    def __init__(self):
        self.alpha = 32 # action precision
        self.beta = 1    # policy precision
        self.eta = 1     # learning rate
        self.tau = 4     # update time constant (gradient descent)
        self.erp = 4     # update reset
        self.chi = 1/64  # Occam window updates
        self.zeta = 3    # Occam window policies

class mdp_layer_options :
    def __init__(self):
        self.update_frequency = 1 # ]0,1] --> At most, you can be updated once every loop, at worst, only once in total

        


class mdp_layer_block_precision :
    def __init__(self,id_,inherited_,correspondance_matrix_=None,prior_=None,to_infer_=None):
        self.id = id_                                               # Denomination 
        self.inherited = inherited_                                 # This block precision is induced by upper states ? (yes / no)
        self.correspondance_matrix = correspondance_matrix_         # SUM(Outcome(lvl+1)*correspondance_matrix) = new_prec
        self.prior = prior_                                         # If we don't depend on a parent node, what is the prior
        self.to_infer = to_infer_                                   # Should we try to infer its value ?

        self.BETA = None                                      # Array, gt values --> Size = (Modality / Factor)  x  T  [-- x Np ? --]
        self.beta = None                                      # Array, infered values --> Size = (Modality / Factor)  x  T  [-- x Np ? --]
        self.beta_n = None                                    # Array, infered values --> Size = Ni  x  (Modality / Factor)  x  T  [-- x Np ? --]
    
    def fill_empty_BETAs(self,fillValue,t):
        assert (isField(self.BETA)),"BETA " + self.id+ "  not implemented ... Please check that precisions are initialized before setting values."
        dim = self.BETA.shape[0] 
        for i in range(dim):
            if (self.BETA[i,t]<0):
                self.BETA[i,t] = fillValue

    def fill_empty_priors(self,fillValue,t):
        assert (isField(self.prior)),"prior " + self.id+ "  not implemented ... Please check that precisions are initialized before setting values."
        dim = self.prior.shape[0] 
        for i in range(dim):
            if (self.prior[i]<0):
                self.prior[i] = fillValue




class mdp_layer_precisions :
    def __init__(self):
        self.policy = mdp_layer_block_precision("POL",False)
        self.A = mdp_layer_block_precision("A",False)
        self.B = mdp_layer_block_precision("B",False)
        self.C = mdp_layer_block_precision("C",False)
        self.D = mdp_layer_block_precision("D",False)
        self.E = mdp_layer_block_precision("E",False)

    def count_inherited_precisions(self):
        count = 0 
        if (self.A.inherited) :
            count = count + 1
        if (self.B.inherited) :
            count = count + 1
        if (self.C.inherited) :
            count = count + 1
        if (self.D.inherited) :
            count = count + 1
        if (self.E.inherited) :
            count = count + 1
        if (self.policy.inherited) :
            count = count + 1
        return count

    def fill_all_empty_BETAs(self,fillValue,t):
        self.policy.fill_empty_BETAs(fillValue,t)
        self.A.fill_empty_BETAs(fillValue,t)
        self.B.fill_empty_BETAs(fillValue,t)
        self.C.fill_empty_BETAs(fillValue,t)
        self.D.fill_empty_BETAs(fillValue,t)
        self.E.fill_empty_BETAs(fillValue,t)

    

class mdp_layer :
    # Agent constructor
    def __init__(self,in_seed = 0):
        self.seed = in_seed

        self.name = '<default name>'
        self.options = mdp_layer_options()
        self.parent = None
        self.child = None
        self.level = 0

        # Simulation parameters
        self.T = 0          # The temporal horizon for this layer (a.k.a, how many timescales it will have to experience)
        self.t = 0          # The current time step, if t==T, the experience is over
        self.Ni = 16         # Inference iterations
        
        
        #Model building blocks -----------------------------------------------------------------------------------------------
        # INPUT Beliefs (learning process, =/= generative process)
        self.a_ = None       # A matrix beliefs (learning)
        self.b_ = None       # B matrix beliefs (learning)
        self.c_ = None       # C matrix beliefs (learning)
        self.d_ = None       # D matrix beliefs (learning)
        self.e_ = None       # E matrix beliefs (learning)

        # INPUT Ground Truth matrices (generative process, will be used in the generative model if no value is provided above)
        self.A_ = None       # A matrix real (generative process)    
        self.B_ = None       # B matrix real (generative process)    
        self.C_ = None       # C matrix real (generative process)    
        self.D_ = None       # D matrix real (generative process)    
        self.E_ = None       # E matrix real (generative process)    
        self.V_ = None       # Allowable policies (T-1 x Np x F)
        self.U_ = None       # Allowable actions (1 x Np x F)
        #--------------------------------------------------------------------------------------------------------------------

        
        self.precisions = mdp_layer_precisions()
              # Precision matrix for correspondance between upper lvl observations and precisions priors / GT values
                             # There should be the same number of outcomes for each modality at lvl (self + 1) as 
                             # the number of weight for precisions.
                             # e.g. : We have 3 outcomes for a given modality (attention for example) in a level n+1 state
                             #        Each outcome will translate to a given precision value (prec_j  = sum_i(O_(n+1)_i,j * BETA_i,j))
                    # TODO : if there are no parents, filling up this dictionnary should provide priors for inference on a monolevel basis

        
        #Optional inputs, might be overriden to force certain states
        self.o = None   # Sequence of observed outcomes
        self.s = None   # Sequence of true states
        self.u = None   # Chosen Action


        #Modulation Parameters
        self.parameters = layer_parameters()
        print(self.parameters.eta)
        
        # Simularion Sizes
        self.Np = 0    #Number of policies
        self.No = []   #Number of outcomes for each MODALITY
        self.Nmod = len(self.No) #Number of modalities
        self.Ns = []   #Number of states for each FACTOR
        self.Nf = len(self.Ns) #Number of factors
        self.Nu = []     # Number of controllable transitions
        
        #OUTPUTS
        
        self.O = None   # Likelihood of observed outcomes
        self.P = []             # Action posterior
        self.R = []             # Policy posterior
        
        self.Q = []             # State posteriors
        self.X = []             # BMA States            
        
        self.G = None     # EFE values
        self.F = None     # VFE values
        self.H = None     # Entropy value i guess ?
        
        self.vn = []            # Neuronal prediction error 
        self.xn = []            # Neuronal encoding of hidden states
        self.pn = []            # Neuronal encoding of policies
        
        

        self.dn = None          # Simulated dopamine response
        self.rt = None          # Simulated reaction times

    
    def initialize_sizes(self):
        """This function initialize the sizes of all blocks"""
        
        
        if (self.Nmod==0):
            self.Nmod = len(self.A_)
        if(self.No ==[]):
            self.No = []
            for i in range(self.Nmod) :
                self.No.append(self.A_[i].shape[0])
        
        if (self.Nf==0):
            self.Nf = len(self.D_)
        if (self.Ns==[]):
            self.Ns = []
            for i in range(self.Nf):
                self.Ns.append(self.D_[i].shape[0])
        
        self.N_induced_precisions = self.precisions.count_inherited_precisions()

        if (self.Nu ==[]):
            if (isField(self.B_)): 
                for f in range(self.Nf) :
                    assert self.B_[f].ndim > 2,"B_ has too little dimensions"
                    self.Nu.append(self.B_[f].shape[2])
                    # B_[f] shoulc always be a >= 3D matrix
            else :
                for f in range(self.Nf) :
                    assert self.b_[f].ndim > 2,"b_ has too little dimensions"
                    self.Nu.append(self.b_[f].shape[2])
                    # b_[f] shoulc always be a >= 3D matrix
        
        if (self.Np==0):
            one_policy=True
            for f in range(self.Nf):
                if(not(self.Nu[f]==1)):
                    one_policy = False
            if (one_policy):
                self.Np = 1
            else :
                assert isField(self.V_),"V_ should be set as model input but has not been."
                self.Np = self.V_.shape[1]

        if (self.Ni < 0) :
            self.Ni = 1


        if(self.name == '') :
            self.name = 'unnamed_model'
    
    def prep_trial(self):       
        self.initialize_sizes()
        Nmod = self.Nmod
        No = self.No
        Nf = self.Nf
        Ns = self.Ns  
        Np = self.Np
        Nu = self.Nu
        Ni = self.Ni
        t = self.t
        T = self.T

        # Use inputs to initialize blocks :
                # Likelihood model a / A
        assert isField(self.A_), "A_ not filled in"
        assert isField(self.B_), "B_ not filled in"


        if isField(self.a_):
            A = normalize(self.A_)

            a = normalize(self.a_)
            a_prior = []
            a_complexity = []
            for modality in range(Nmod):
                a_prior.append(np.copy(self.a_[modality]))
                a_complexity.append( spm_wnorm(a_prior[modality])*(a_prior[modality]>0) )
        elif isField(self.A_) :
            A = normalize(self.A_)

            a = normalize(self.A_)
        else :
            raise RuntimeError("- No perception matrix A as input.")
        
 
        # Transition model b / B
        if isField(self.b_): # If we are learning the transition matrices
            b = normalize(self.b_)
            
            b_prior = []
            b_complexity = []
            for factor in range(Nf):   # For all factors
                b_prior.append(np.copy(self.b_[factor]))
                b_complexity.append(spm_wnorm(b_prior[factor])*(b_prior[factor]>0))
        elif isField(self.B_) :
            B = normalize(self.B_)

            b = normalize(self.B_)
        else :
            raise RuntimeError("- No transition matrix B as input.")

        
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

        if isField(self.V_):
            V = self.V_.astype(np.int)
        else : 
            assert (Np == 1), "If V_ is not input, there should only be one policy available .."
            V = np.zeros((T,1,Nf))

        # --------------------------------------------------------
        self.a = a
        self.A = A
        if isField(self.a_):
            self.a_prior = a_prior
            self.a_complexity = a_complexity
        
        self.b = b
        self.B = B
        if isField(self.b_):
            self.b_prior = b_prior
            self.b_complexity = b_complexity
        
        self.C = C
        if isField(self.c_):
            self.c_prior = c_prior
        
        self.d = d
        self.D = D
        if isField(self.d_):
            self.d_prior = d_prior
            self.d_complexity = d_complexity
        
        self.E = E

        self.V = V
        # --------------------------------------------------------

        # the value inside observations/ true states  can be:
        # <0 if not yet defined : it will be created by the generative process
        # x in [0,No[modality]-1]/[0,Ns[factor]-1] for a given modality/factor at a given time if we want to explicit some value

        #OUTCOMES -------------------------------------------------------------
        self.O = []
        for modality in range(Nmod):
            self.O.append(np.zeros((No[modality],T)))                                          
        o = np.full((Nmod,T),-1)
                                    
        if isField(self.o):  # There is an "o" matrix with fixed values ?
            o[self.o >=0] = self.o[self.o>=0]
            # If there are fixed values for the observations, they will be copied
        self.all_outcomes_input = np.sum(o<0)==0  # If we have all outcomes as input, no need to infer them 
        self.o = np.copy(o)

        #STATES ------------------------------------------------------------------
        # true states 
        s = np.full((Nf,T),-1) 
        if isField(self.s):
            s[self.s >=0] = self.s[self.s >=0]
        self.all_states_input = np.sum(s<0)==0  # If we have all states as input, no need to infer them 
        self.s = np.copy(s)
        
        # Posterior expectations of hidden states
        # state posteriors
        self.x = []
        self.xn = []
        self.X = []
        self.X_archive = []
        for f in range(Nf):
            self.x.append(np.zeros((Ns[f],T,Np)) + 1./Ns[f])                     # Posterior expectation of all hidden states depending on each policy
            self.xn.append(np.zeros((Ni,Ns[f],T,T,Np)) + 1./Ns[f])
            self.X.append(np.tile(np.reshape(d[f],(-1,1)),(1,T)))                # Posterior expectation of all hiddent states given posterior on policy
            self.X_archive.append(np.tile(np.reshape(d[f],(-1,1,1)),(T,T)))      # Estimation at time t of BMA states at time tau
            for k in range(Np):
                self.x[f][:,0,k] = d[f]

        self.vn = []
        for f in range(Nf):        
            self.vn.append(np.zeros((Ni,Ns[f],T,T,Np)))  # Recorded neuronal prediction error
        
        #ACTIONS ------------------------------------------------------------------

        # >>> Action_selection

        #history of posterior over action
        self.u_posterior_n = np.zeros((Np,Ni*T))             
        #posterior over action
        self.u_posterior = np.zeros(tuple(Nu)+(T-1,))                
        
        # >>> Chosen Action

        u_temp = np.full((Nf,T-1),-1)  
        if isField(self.u):
            u_temp[self.u>=0] = self.u[self.u>=0]
        self.u = u_temp

        #POLICIES ------------------------------------------------------------------
        #history of posterior over policy
        self.p_posterior_n = np.zeros((Np,Ni*T))             
        #posterior over policy
        self.p_posterior = np.zeros((Np,T))                 
        if (Np == 1) :
            self.p_posterior = np.ones((Np,T))

        
              
        # Allowable policies
        p = np.zeros((Np,))
        for policy in range(Np): # Indices of allowable policies
            p[policy] = policy
        self.p = p.astype(np.int)

        # -------------------------------------------------------------
        #TODO : Initialize output variables for a run
        self.L = []


        self.F = np.zeros((self.Np,self.T))
        self.G = np.zeros((self.Np,self.T))
        self.H = np.zeros((self.T,))     
        
        self.wn = None          # Neuronal encoding of policy precision
        self.dn = None          # Simulated dopamine response
        self.rt = np.zeros((T,))          # Simulated reaction times
        

        def init_precisions():
            # Initialize precisions
            beta_mat = np.full((1,T),-1.0)
            if isField(self.precisions.policy.BETA) :
                beta_mat[self.precisions.policy.BETA>0] = self.precisions.policy.BETA[self.precisions.policy.BETA>0]
            self.precisions.policy.BETA = np.copy(beta_mat)
            prior_mat =  np.full((1,),-1.0)
            if isField(self.precisions.policy.prior) :
                prior_mat[self.precisions.policy.prior>0] = self.precisions.policy.prior[self.precisions.policy.prior>0]
            self.precisions.policy.prior = np.copy(prior_mat)
            self.precisions.policy.beta = np.ones((1,T))
            self.precisions.policy.beta_n = np.ones((1,Ni*T))
            
            

            beta_mat = np.full((Nmod,T),-1.0)
            if isField(self.precisions.A.BETA) :
                beta_mat[self.precisions.A.BETA>0] = self.precisions.A.BETA[self.precisions.A.BETA>0]
            self.precisions.A.BETA = np.copy(beta_mat)
            prior_mat =  np.full((Nmod,),-1.0)
            if isField(self.precisions.A.prior) :
                prior_mat[self.precisions.A.prior>0] = self.precisions.A.prior[self.precisions.A.prior>0]
            self.precisions.A.prior = np.copy(prior_mat)
            self.precisions.A.beta = np.ones((Nmod,T))
            self.precisions.A.beta_n = np.ones((Nmod,Ni*T))


            beta_mat = np.full((Nf,T),-1.0)
            if isField(self.precisions.B.BETA) :
                beta_mat[self.precisions.B.BETA>0] = self.precisions.B.BETA[self.precisions.B.BETA>0]
            self.precisions.B.BETA = np.copy(beta_mat)
            prior_mat =  np.full((Nf,),-1.0)
            if isField(self.precisions.B.prior) :
                prior_mat[self.precisions.B.prior>0] = self.precisions.B.prior[self.precisions.B.prior>0]
            self.precisions.B.prior = np.copy(prior_mat)
            self.precisions.B.beta = np.ones((Nf,T))
            self.precisions.B.beta_n = np.ones((Nf,Ni*T))


            beta_mat = np.full((Nmod,T),-1.0)
            if isField(self.precisions.C.BETA) :
                beta_mat[self.precisions.C.BETA>0] = self.precisions.C.BETA[self.precisions.C.BETA>0]
            self.precisions.C.BETA = np.copy(beta_mat)
            prior_mat =  np.full((Nmod,),-1.0)
            if isField(self.precisions.C.prior) :
                prior_mat[self.precisions.C.prior>0] = self.precisions.C.prior[self.precisions.C.prior>0]
            self.precisions.C.prior = np.copy(prior_mat)
            self.precisions.C.beta = np.ones((Nmod,T))
            self.precisions.C.beta_n = np.ones((Nmod,Ni*T))


            beta_mat = np.full((Nf,T),-1.0)
            if isField(self.precisions.D.BETA) :
                beta_mat[self.precisions.D.BETA>0] = self.precisions.D.BETA[self.precisions.D.BETA>0]
            self.precisions.D.BETA = np.copy(beta_mat)
            prior_mat =  np.full((Nf,),-1.0)
            if isField(self.precisions.D.prior) :
                prior_mat[self.precisions.D.prior>0] = self.precisions.D.prior[self.precisions.D.prior>0]
            self.precisions.D.prior = np.copy(prior_mat)
            self.precisions.D.beta = np.ones((Nf,T))
            self.precisions.D.beta_n = np.ones((Nf,Ni*T))


            beta_mat = np.full((1,T),-1.0)
            if isField(self.precisions.E.BETA) :
                beta_mat[self.precisions.E.BETA>0] = self.precisions.E.BETA[self.precisions.E.BETA>0]
            self.precisions.E.BETA = np.copy(beta_mat)
            prior_mat =  np.full((1,),-1.0)
            if isField(self.precisions.E.prior) :
                prior_mat[self.precisions.E.prior>0] = self.precisions.E.prior[self.precisions.E.prior>0]
            self.precisions.E.prior = np.copy(prior_mat)
            self.precisions.E.BETA = np.ones((1,T))
            self.precisions.E.beta = np.ones((1,T))
            self.precisions.E.beta_n = np.ones((1,Ni*T))
        init_precisions()

    def get_ground_truth_values(self):
        """ 
        Generative process : we create here the values used during the inference process.
        GT values are comprised of :
        - observations
        - precisions (if the precisions are dependent on upper states)
        """
        Nmod = self.Nmod
        No = self.No
        Nf = self.Nf
        Ns = self.Ns  
        Np = self.Np
        Nu = self.Nu
        Ni = self.Ni
        t = self.t
        T = self.T
        msg = "Gathering GT values at time "  + str(self.t+1) +" / " + str(self.T) + " (layer " + str(self.level) + " ) ..."
        #print(msg,end=' ')

        # Fetch precisions :  ---------------------------------------------------------------------------------------------------------------------------------------
        policy_precision = 1.0              # Prior for policy selection
        A_precision = np.ones((Nmod,))  # One perceptual precision per modality
        B_precision = np.ones((Nf,))    # One transition precision per factor

        # TODO : study the following and their eventual precision
        C_precision = np.ones((Nmod,))                
        D_precision = np.ones((Nf,))
        E_precision = 1.0

        # If we have a parent, we might use its observations to infer our precisions :
        if (self.parent != None):
            try :
                upper_level_observations = self.parent.O 
            except :
                raise RuntimeError("Parent observations should be available before fetching precisions (level : "  + str(self.level) + " ).")           
            # If everything is ok, there should be as many modailities in upper_level_observations as precision_weighted
            # functions in this layer (--> Nmod(layer + 1) = len([A,B,policy])) if we only modulate A,B and policy
            error_message = "/!\ Error : O(" + str(self.level + 1) + ") should have " + str(self.N_induced_precisions) + " modalities but it has " + str(len(upper_level_observations)) +" ."
            error_message += "\n       -- Error recorded at level " + str(self.level) + "  at time " + str(self.t) + " / " + str(self.T)
            assert(len(upper_level_observations) ==  self.N_induced_precisions),error_message


            if(self.precisions.policy.inherited) :
                # The precision is given by the upper states
                raise NotImplementedError("Precision on policy selection not implemented")
                if (self.precisions.policy.BETA[0,t]<0):
                    print('Here i would calculate policy.BETA[0,t] using both the o(lvl+1) and BETA matrixes')
            else :
                # We have a parent, but it does not define our precisions
                print("Parent does not define precisions on policy selection")


            if(self.precisions.A.inherited) :
                # The precision is given by the upper states
                raise NotImplementedError("Precision on A matrices not implemented")
                for mod in range(Nmod):
                    if (self.precisions.A.BETA[mod,t]<0):
                        print('Here i would calculate A.BETA[mod,t] using both the o(lvl+1) and BETA matrixes')
            else :
                # We have a parent, but it does not define our precisions
                print("Parent does not define precisions on A matrix")

            
            if(self.precisions.B.inherited) :
                # The precision is given by the upper states
                raise NotImplementedError("Precision on B matrices not implemented")
                for f in range(Nf):
                    if (self.precisions.B.BETA[f,t]<0):
                        print('Here i would calculate B.BETA[f,t] using both the o(lvl+1) and BETA matrixes')
            else :
                # We have a parent, but it does not define our precisions
                print("Parent does not define precisions on B matrix")


            if(self.precisions.C.inherited) :
                # The precision is given by the upper states
                raise NotImplementedError("Precision on C matrices not implemented")
                for mod in range(Nmod):
                    if (self.precisions.C.BETA[mod,t]<0):
                        print('Here i would calculate C.BETA[mod,t] using both the o(lvl+1) and BETA matrixes')
            else :
                # We have a parent, but it does not define our precisions
                print("Parent does not define precisions on C matrix")


            if(self.precisions.D.inherited) :
                # The precision is given by the upper states
                raise NotImplementedError("Precision on D matrices not implemented")
                for f in range(Nf):
                    if (self.precisions.D.BETA[f,t]<0):
                        print('Here i would calculate D.BETA[f,t] using both the o(lvl+1) and BETA matrixes')
            else :
                # We have a parent, but it does not define our precisions
                print("Parent does not define precisions on D matrix")


            if(self.precisions.E.inherited) :
                # The precision is given by the upper states
                raise NotImplementedError("Precision on E matrices not implemented")
                if (self.precisions.policy.BETA[0,t]<0):
                    print('Here i would calculate E.BETA[0,t] using both the o(lvl+1) and BETA matrixes')
            else :
                # We have a parent, but it does not define our precisions
                print("Parent does not define precisions on E matrix")
        self.precisions.fill_all_empty_BETAs(1.0,t)



        # TODO :the GT matrices must be modulated by precision values 

        # Fetch outcomes :  ---------------------------------------------------------------------------------------------------------------------------------------
        # If we already have all the outcomes, we don't need to use the architecture to infer internal states:
        if (self.all_outcomes_input) :
            # self.o contains all the outcomes for the trial, therefore we don't need to perform trickedown using states. 
            print("We already have a complete observation sequence :  no need to generate it.")
        else :
            # We're missing all or part of the outcomes. To generate those, we need to simulate ground truth hidden states.

            # If those hidden states are ALL already known as input, no need to infer them. They are all present in self.s
            # If even one is missing, we use this method to compute it. 
            if not(self.all_states_input) :

                # true states definition.
                for f in range(Nf) :
                #        Here we sample from the prior distribution over states to obtain the
                #        % state at each time point. At T = 1 we sample from the D vector, and at
                #        % time T > 1 we sample from the B matrix. To do this we make a vector 
                #        % containing the cumulative sum of the columns (which we know sum to one), 
                #        % generate a random number (0-1),and then use the find function to take 
                #        % the first number in the cumulative sum vector that is >= the random number. 
                #        % For example if our D vector is [.5 .5] 50% of the time the element of the 
                #        % vector corresponding to the state one will be >= to the random number. 
                    #Dbar = precision_weight(A,D_precision)
                    if (self.s[f,t] < 0) :
                        if (t==0) :
                            prob_state = self.D[f]
                        else :
                            prob_state = self.B[f][:,self.s[f,t-1],self.u[f,t-1]]
                        self.s[f,t] = np.argwhere(r.random() <= np.cumsum(prob_state,axis=0))[0]
                self.s[:,t] = self.s[:,t].astype(np.int)
            else :
                print("We already have a complete state sequence :  no need to generate it.")
            
            #True outcome definition
            for modality in range(Nmod):
                if (self.o[modality,t] < 0):
                    ind = (slice(None),) + tuple(self.s[:,t])  # Indice corresponding to the current active states
                    self.o[modality,t] = np.argwhere(r.random() <= np.cumsum(self.A[modality][ind],axis=0))[0]
            self.o[:,t] = self.o[:,t].astype(np.int)

            # Get probabilistic outcomes from samples
            O = self.O
            for modality in range(Nmod):
                vec = np.zeros((1,No[modality]))
                #print(vec.shape)
                index = self.o[modality,t]
                #print(index)
                vec[0,index] = 1
                O[modality][:,t] = vec
            self.O = O

        # ************************* True states and outcomes for this t are generated. **********************************************

        # END OF GENERATIVE PROCESS !!
        #-------------------------------------------------------------------------------------------------
        #print("Success !")

    def perform_state_inference(self):
        Nmod = self.Nmod
        No = self.No

        Nf = self.Nf
        Ns = self.Ns 

        Np = self.Np
        Nu = self.Nu

        Ni = self.Ni
        t = self.t
        T = self.T
        t = self.t


        intro_msg = "Infering hidden states at time "  + str(t+1) +" / " + str(T) + " (layer " + str(self.level) + " ) ..."
        #print(intro_msg,end=' ' ) #'\n')



        #Posterior predictive density over hidden external states
        xq = []
        xqq = []
        for f in range(self.Nf) :
            if (t ==0) :
                xqq.append(self.X[f][:,t])
            else :
                xqq.append(np.dot(np.squeeze(self.b[f][:,:,self.u[f,t-1]]),self.X[f][:,t-1]))
            xq.append(self.X[f][:,t])

        # Likelihood of hidden states
        self.L.append(1)
        for modality in range (Nmod):
            self.L[t] = self.L[t] * spm_dot(self.a[modality],self.O[modality][:,t])

        # Policy reduction if too unlikely --> TODO : Place it at the end of the tick ?
        if (isField(self.parameters.zeta)):
                if not(isField(self.U_)) and (t>0):
                    F = nat_log(self.p_posterior[self.p,t-1])
                    self.p = self.p[(F-np.max(F))>-self.parameters.zeta]
        
                        
        tstart = time.time()
        for f in range(Nf):
            self.x[f] = softmax(nat_log(self.x[f])/self.parameters.erp,axis = 0,center = False)
        
        # % Variational updates (hidden states) under sequential policies
        #%==============================================================
        S = self.V.shape[0] + 1
        if (self.U_):
            R = t
        else :
            R = S
        F = np.zeros((Np,))
        G = np.zeros((Np,))

        for policy in self.p :
            dF = 1 # Criterion for given policy
            for iteration in range(Ni) :
                
                F[policy] = 0
                
                for tau in range(S): #Loop over future time points
                    
                    #posterior over outcomes
                    if (tau <= t) :
                        for factor in range(Nf):
                            xq[factor]=np.copy(self.x[factor][:,tau,policy])
                    for factor in range (Nf):
                        #  hidden state for this time and policy
                        sx = np.copy(self.x[factor][:,tau,policy])
                        qL = np.zeros((Ns[factor],))
                        v = np.zeros((Ns[factor],))
                        
                        # evaluate free energy and gradients (v = dFdx)
                        if ((dF > np.exp(-8)) or (iteration > 3)) :
                            
                            # marginal likelihood over outcome factors
                            if (tau <= t) :
                                qL = spm_dot(self.L[tau],xq,factor)
                                qL = nat_log(qL)
                            qx = nat_log(sx)
                            
                            
                            #Empirical priors (forward messages)
                            if (tau == 0):
                                px = nat_log(self.d[factor])
                            else :
                                px = nat_log(np.dot(np.squeeze(self.b[factor][:,:,self.V[tau-1,policy,factor]]),self.x[factor][:,tau-1,policy]))
                            v = v +px + qL - qx
                            #Empirical priors (backward messages)
                            if (tau == R-1) :
                                px = 0
                            else : 
                                px = nat_log(np.dot(normalize(self.b[factor][:,:,self.V[tau,policy,factor]].T),self.x[factor][:,tau+1,policy]))
                                v = v +px + qL - qx
                            
                            if ((tau==0) or (tau==S-1)):
                                F[policy] = F[policy] + 0.5*np.dot(sx.T,v)
                            else :
                                F[policy] = F[policy] + np.dot(sx.T,0.5*v - (Nf-1)*qL/Nf)
                                
                            v = v - np.mean(v)
                            sx = softmax(qx + v/self.parameters.tau)

                        else :
                            F[policy] = G[policy] # End of condition
                            
                        self.x[factor][:,tau,policy] = sx
                        xq[factor] = np.copy(sx)
                        self.xn[factor][iteration,:,tau,t,policy] = sx
                        self.vn[factor][iteration,:,tau,t,policy] = v
                        
                # end of loop onn tau --> convergence :
                if (iteration > 0):
                    dF = F[policy] - G[policy]
                G = np.copy(F)

        self.F[:,t] = F
        #print("Success !")
        return time.time() - tstart

    def calculate_EFE(self):
        """Using our current estimation of hidden states, try to estimate the value of our exxpected free energy depending on policy. 
            Of course if we only have one policy, the result is useless"""
        Q = np.zeros((self.Np,)) # Actual EFE
        S = self.V.shape[0] + 1

        # EFE Calculation
        for policy in self.p:
            # Bayesian surprise about initial conditions
            if isField(self.d_):
                for factor in range (self.Nf):
                    Q[policy] = Q[policy] - spm_dot(self.d_complexity[factor],self.x[factor][:,0,policy])

            for timestep in range(self.t,S):
                xq = []
                for factor in range (self.Nf):
                    xq.append(self.x[factor][:,timestep,policy])
                
                #Bayesian surprise about states
                Q[policy] = Q[policy] + G_epistemic_value(self.a,xq) 
                for modality in range(self.Nmod):
                    
                    #Prior preferences about outcomes
                    qo = spm_dot(self.a[modality],xq)   #predictive observation posterior
                    Q[policy] = Q[policy] + np.dot(qo.T,self.C[modality][:,timestep])
                    #Bayesian surprise about parameters
                    if isField(self.a_):
                        Q[policy] = Q[policy] - spm_dot(self.a_complexity[modality],[qo]  + xq[:])
                #[predictive_observation_posterior] + Expected_states[:])
                #print(Q[policy])
                #End of loop on policies                                
        self.G[:,self.t] = Q

    
    def perform_policy_inference(self):
        #print("I perform policy inference using the infered states")

        Nmod = self.Nmod
        No = self.No

        Nf = self.Nf
        Ns = self.Ns 

        Np = self.Np
        Nu = self.Nu

        Ni = self.Ni
        t = self.t
        T = self.T
        t = self.t
        p = self.p

        tstart = time.time()
        print(self.E)
        print(self.F)
        print(self.Q)
        if (Np>1) :
            self.calculate_EFE()



            Q = self.G[:,self.t]

            # PRECISION INFERENCE
            if (t>0):
                beta = self.precisions.policy.BETA[0,t-1]
            else :
                beta = self.precisions.policy.prior[0]
            qb = beta
            print(t,beta)
            gamma_t = 1/beta

            
            for iteration in range(Ni):
                # posterior and prior beliefs about policies
                q_p = softmax(nat_log(self.E)[p] + gamma_t*Q[p] + self.F[p,t])      # Posterior over policies
                p_p = softmax(nat_log(self.E)[p] + gamma_t*Q[p])             # Prior over policies

                if (self.precisions.policy.to_infer):
                    eg = np.dot((q_p-p_p).T,Q[p])                           # Affective Charge (C. Hesp 2021)
                    dFdg = qb - self.precisions.policy.prior[0] + eg 
                    qb = qb - dFdg/2.                                       # update of beta posterior
                    #qb = qb - (qb - beta_prior + eg
                    gamma_t = 1/qb                                          # update of gamma posterior
                else :
                    gamma_t = 1/beta

                #dopamine responses
                n = t*Ni + iteration
                self.precisions.policy.beta_n[0,n] = 1/gamma_t
                self.p_posterior_n[p,n] = q_p
            
            self.p_posterior[p,t] = q_p
            
            self.precisions.policy.beta[0,t] = 1/gamma_t         
            self.precisions.policy.BETA[0,t] = 1/gamma_t
            print("---------------------------------")
            print(self.F[p,t])
            print("---------------------------------")


        #BMA states calculation (given the posterior over policies, what do our hidden states look like ?)
        S = self.V.shape[0] + 1
        for factor in range(Nf):
            for tau in range(S):
                self.X[factor][:,tau] =np.dot(self.x[factor][:,tau,:],self.p_posterior[:,t])    # BMA state = posterior over state depending on policy chosen * posterior over policy
                self.X_archive[factor][:,t,tau] = self.X[factor][:,tau]                  # Let's stock its value
        policy_inference_time = time.time() - tstart

        if (Np > 1) :
            self.H[t] = np.dot(q_p.T,self.F[p,t]) - np.dot(q_p.T,(nat_log(q_p) - nat_log(p_p)))
        else :
            self.H[t] = self.F[p,t] # - (nat_log(q_p) - nat_log(p_p))  --> = 0 ?
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

        return policy_inference_time

    def perform_action_selection(self):
        Nmod = self.Nmod
        No = self.No

        Nf = self.Nf
        Ns = self.Ns 

        Np = self.Np
        Nu = self.Nu

        Ni = self.Ni
        t = self.t
        T = self.T
        t = self.t
        tstart = time.time()

        # Action selection !!
        if (t<T-1):
            #Marginal posterior over action
            u_posterior_intermediate = np.zeros(tuple(Nu))  # Action posterior intermediate
            
            for policy in range(Np):
                
                sub = self.V[t,policy,:] # coordinates for the corresponding action wrt t and policy
                action_coordinate = tuple(sub)
                u_posterior_intermediate[action_coordinate] = u_posterior_intermediate[action_coordinate] + self.p_posterior[policy,t]
            
            u_posterior_intermediate = softmax(self.parameters.alpha*nat_log(u_posterior_intermediate)) # randomness in action selection --> Low alpha = random
            self.u_posterior[...,t] = u_posterior_intermediate
            #action_posterior[...,t] = action_posterior_intermediate
            # Next action : sampled from marginal posterior
            for factor in range(Nf):
                if (self.u[factor,t]<0) : # The choice of action is not overriden
                    if(Nu[factor]>1) :
                        randomfloat = r.random()
                        #print(np.cumsum(action_posterior_intermediate,axis=factor))
                        ind = np.argwhere(randomfloat <= np.cumsum(u_posterior_intermediate,axis=factor))[0]
                        #ind est de taille n où n est le nombre de transitions à faire :
                        self.u[factor,t] = ind[factor]
                        #print(b, np.cumsum(action_posterior_intermediate,axis=1),ind[0])                
                    else :
                        self.u[factor,t] = 0
            
            if isField(self.U_) :
                for factor in range(Nf):
                    self.V[t,:,factor] = self.u[factor,t]
                
                for j in range (self.U_.shape[0]) :
                    if (t+1 < T-1) :
                        self.V[t+1,:,:] = self.U_[:,:]
                    
                for factor in range(Nf):
                    for policy in range (Np):
                        self.x[factor][:,:,policy] = 1.0/Ns[factor]
            #End of condition on U_
        # End of Action selection

        if (t==T-1): #Accumulate all evidences
            if (T==1):
                self.u = np.zeros((Nf,1)) 
        return time.time()- tstart

    def perform_precision_inference(self):
        """ Given the outcomes / states infered, let's try to figure out which precisions are the most likely ! It will in turn allow us to predict the higher lvl outcomes"""
        print("I will infer precisions")

    def tick(self):
        """ The layer gathers all data from parent & children states and performs inference"""
        t = self.t
        T = self.T 

        if(t<T):
            # print(np.round(self.x[0],2))
            self.get_ground_truth_values()
            state_inference_time = self.perform_state_inference()          
            policy_inference_time = self.perform_policy_inference()
            action_selection_time = self.perform_action_selection()




            self.rt[t] = state_inference_time + policy_inference_time
            self.t = self.t + 1
        else :
            print("No actions were conducted this round (time exceeded trial horizon)")

    def learn_from_experience(self):
        print("Wow this was insightful : i'm gonna learn from that !")
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
        #                print('-----------------')
        #                print(u[:,t])
                        
                for factor in range(Nf):
        #                    print(np.round(x[factor][:,t,:],2))
        #                    print(np.round(x[factor][:,t-1,:],2))
        #                    print('*')
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
        
    def postrun(self) :
        self.learn_from_experience()

    def run(self,verbose=0):
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
        #                print('-----------------')
        #                print(u[:,t])
                        
                for factor in range(Nf):
        #                    print(np.round(x[factor][:,t,:],2))
        #                    print(np.round(x[factor][:,t-1,:],2))
        #                    print('*')
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
        self.archive = X_archive # BMA States estimation at all times
        self.ran = True

if __name__ == "__main__":
    eem = explore_exploit_model(0.8)
    eem.name = "Basic model"

    layer = mdp_layer(eem.seed)
    layer.name = "Layer model"
    layer.A_ = eem.A_
    layer.B_ = eem.B_
    layer.D_ = eem.D_
    layer.d_ = eem.d_
    layer.C_ = eem.C_
    layer.T = 3
    layer.V_ = eem.V_
    layer.precisions.policy.prior = np.array([1.0])
    layer.precisions.policy.to_infer = True


    print("----------------------------------------------")
    print("        MODEL " + str(layer.name) )
    print()
    print("seed: " + str(layer.seed))
    print("----------------------------------------------")
        
    layer.prep_trial()
    for i in range(layer.T):
        layer.tick()

    # eem.run()

    # print(layer.s)
    # print(eem.s)
    # print("-------")
    # print(layer.o)
    # print(eem.o)
    # print("-------")
    # print(layer.F)
    # print(eem.F)
    # print("-------")
    # print(layer.G)
    # print(eem.G)
    # print("-------")

    # k = layer.T*layer.Ni
    # x = np.linspace(0,k,k)
    # plt.plot(x,1.0/layer.precisions.policy.beta_n[0])
    # plt.plot(x,eem.wn)
    # plt.show()