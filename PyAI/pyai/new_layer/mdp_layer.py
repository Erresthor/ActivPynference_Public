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
from enum import Enum
import random as r
import time 
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace

from .spm_forwards import spm_forwards
from .spm_backwards import spm_backwards
from .layer_prep import prep_layer
from .layer_precisions import *
from .layer_learn import *

from ..base.function_toolbox import normalize,softmax,nat_log,precision_weight
from ..base.function_toolbox import spm_wnorm,cell_md_dot,md_dot, spm_cross,spm_KL_dir,spm_psi, spm_dot
from ..base.function_toolbox import G_epistemic_value
from ..base.function_toolbox import spm_kron,spm_margin,spm_dekron
from ..base.miscellaneous_toolbox import isField, isNone
from ..base.miscellaneous_toolbox import flexible_copy

class hyperparameters :
    def __init__(self):
        self.alpha = 32 # action precision
        self.beta = 1    # policy precision
        self.eta = 1     # learning rate
        self.tau = 4     # update time constant (gradient descent)
        self.erp = 4     # update reset
        self.chi = 1/64  # Occam window updates
        self.zeta = 3    # Occam window policies

class mdp_layer_learnoptions :
    def __init__(self):
        self.T_horizon = 1
        self.update_frequency = 1 # ]0,1] --> At most, you can be updated once every loop, at worst, only once in total
                                    # To be implemented
        
        self.learn_during_trial = False
        self.memloss = 0

        self.learn_a = True
        self.learn_b = True
        self.learn_c = False
        self.learn_d = True
        self.learn_e = False

class process_layer :
    def __init__(self):
        # INPUT Ground Truth matrices (generative process, will be used in the generative model if no value is provided above) --> What actually happens
        self.A_ = None       # A matrix real (generative process)    
        self.B_ = None       # B matrix real (generative process)    
        self.C_ = None       # C matrix real (generative process)    
        self.D_ = None       # D matrix real (generative process)    
        self.E_ = None       # E matrix real (generative process) 

        self.policy_method = Policy_method.UNDEFINED
        self.V_ = None       # Allowable policies (T-1 x Np x F)
        self.U_ = None       # Allowable actions (1 x Np x F)

        self.o_ = None   # IMPOSED Sequence of observed outcomes (input by exp)
        self.s_ = None   # IMPOSED Sequence of true states (input by exp)
        self.u_ = None   # IMPOSED Chosen Actions (input by exp)

class model_layer :
    # Agent constructor
    def __init__(self,in_seed = 0):
        self.name = 'default name'
        self.verbose = False

        self.seed = in_seed

        self.hyperparams = hyperparameters()
        self.learn_options = mdp_layer_learnoptions()

        #Model building blocks -----------------------------------------------------------------------------------------------
        # INPUT Beliefs (learning process, =/= generative process) --> What our agent believes about the dynamics of the world
        self.a_ = None       # A matrix beliefs (learning)
        self.b_ = None       # B matrix beliefs (learning)
        self.c_ = None       # C matrix beliefs (learning)
        self.d_ = None       # D matrix beliefs (learning)
        self.e_ = None       # E matrix beliefs (learning)

        # Simulation parameters
        self.T = 0          # The temporal horizon for this layer (a.k.a, how many timescales it will have to experience)
        self.t = 0          # The current time step, if t==T, the experience is over        

        self.o = None   # Sequence of observed outcomes
        self.s = None   # Sequence of true states
        self.u = None   # Chosen Actions     
        self.K = None   # Chosen action combination index

        self.reinit()
    
    def reinit(self):
        self.t = 0

                #Optional inputs, might be overriden to force certain states
        #self.o = None   # Sequence of observed outcomes
        #self.s = None   # Sequence of true states
        #self.u = None   # Chosen Actions
        #self.K = None   # Chosen action combination index


        #Modulation Parameters
        self.parameters = layer_parameters()
        
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

        self.u = None

        self.Q = []             # State posteriors
        self.X = []             # BMA States        
        self.S = []    
        
        self.G = None     # EFE values
        self.F = None     # VFE values
        self.H = None     # Entropy value i guess ?
        
        self.vn = []            # Neuronal prediction error 
        self.xn = []            # Neuronal encoding of hidden states
        self.pn = []            # Neuronal encoding of policies
        
        self.dn = None          # Simulated dopamine response
        self.rt = None          # Simulated reaction times

        self.o = None   # Sequence of observed outcomes
        self.s = None   # Sequence of true states
        self.u = None   # Chosen Actions

        self.K = None   # Chosen action combination index
        
        
        self.w = None # Posterior belief precision



        # Free energies container for model elements
        self.FE_dict = {
            'Fa':[],
            'Fb':[],
            'Fc':[],
            'Fd':[],
            'Fe':[]
        }

    def copy(self,keep_u = False):
        copied_lay = mdp_layer()

        copied_lay.verbose = self.verbose
        copied_lay.seed = self.seed

        copied_lay.name = self.name + '- copy'
        copied_lay.options = self.options

        copied_lay.T  = self.T
        
        if not(keep_u):
            copied_lay.u = None
        
        copied_lay.a_ = flexible_copy(self.a_)
        copied_lay.A_ = flexible_copy(self.A_)
        copied_lay.b_ = flexible_copy(self.b_)
        copied_lay.B_ = flexible_copy(self.B_)
        copied_lay.c_ = flexible_copy(self.c_)
        copied_lay.C_ = flexible_copy(self.C_)
        copied_lay.d_ = flexible_copy(self.d_)
        copied_lay.D_ = flexible_copy(self.D_)
        copied_lay.e_ = flexible_copy(self.e_)
        copied_lay.E_ = flexible_copy(self.E_)

        copied_lay.o = flexible_copy(self.o)
        copied_lay.s = flexible_copy(self.s)

        copied_lay.policy_method = Policy_method.UNDEFINED
        copied_lay.V_ = flexible_copy(self.V_)       # Allowable policies (T-1 x Np x F)
        copied_lay.U_ = flexible_copy(self.U_)       # Allowable actions (1 x Np x F)

        copied_lay.precisions = self.precisions
        copied_lay.reinit()

        return copied_lay

    def prep_trial(self):
        self.reinit()
        prep_layer(self)

    def get_ground_truth_values(self):
        """ 
        Generative process : we create here the values used during the inference process.
        Various methods used : mirror MDP if no value input / using the input values if they exist
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
        Ni = self.options.Ni
        t = self.t
        T = self.T
        # msg = "Gathering GT values at time "  + str(self.t+1) +" / " + str(self.T) + " (layer " + str(self.level) + " ) ..."
        # print(msg,end=' ')

        # Fetch precisions :  ---------------------------------------------------------------------------------------------------------------------------------------
        policy_precision = 1.0              # Prior for policy selection
        A_precision = np.ones((Nmod,))  # One perceptual precision per modality
        B_precision = np.ones((Nf,))    # One transition precision per factor

        # TODO : study the following and their eventual precision
        C_precision = np.ones((Nmod,))                
        D_precision = np.ones((Nf,))
        E_precision = 1.0 

        #print("-----------")
        # If we have a parent, we might use its observations to infer our precisions :
        if (self.parent != None):
            try :
                upper_level_observations = self.parent.o 
            except :
                raise RuntimeError("Parent observations should be available before fetching precisions (level : "  + str(self.level) + " ).")           
            # If everything is ok, there should be as many modailities in upper_level_observations as precision_weighted
            # functions in this layer (--> Nmod(layer + 1) = len([A,B,policy])) if we only modulate A,B and policy
            error_message = "/!\ Error : O(" + str(self.level + 1) + ") should have " + str(self.N_induced_precisions) + " modalities but it has " + str(len(upper_level_observations)) +" ."
            error_message += "\n       -- Error recorded at level " + str(self.level) + "  at time " + str(self.t) + " / " + str(self.T)
            assert(len(upper_level_observations) ==  self.N_induced_precisions),error_message

            coordinates = tuple(upper_level_observations[:,t])
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
                for mod in range(Nmod):
                    if (self.precisions.A.BETA[mod,t]<0):
                        # precision (len = Nmod) = Correspondance_matrix x O_1 x O_2 x ... x O_Nmod
                        precision_coord = (mod,) + coordinates
                        self.precisions.A.BETA[mod,t] = self.precisions.A.correspondance_matrix[precision_coord]
            else :
                # We have a parent, but it does not define our precisions
                print("Parent does not define precisions on A matrix")

            
            if(self.precisions.B.inherited) :
                # The precision is given by the upper states
                for f in range(Nf):
                    if (self.precisions.B.BETA[f,t]<0):
                        precision_coord = (f,) + coordinates
                        self.precisions.B.BETA[f,t] = self.precisions.B.correspondance_matrix[precision_coord]
            else :
                # We have a parent, but it does not define our precisions
                print("Parent does not define precisions on B matrix")


            if(self.precisions.C.inherited) :
                # The precision is given by the upper states
                for mod in range(Nmod):
                    if (self.precisions.C.BETA[mod,t]<0):
                        precision_coord = (mod,) + coordinates
                        self.precisions.C.BETA[mod,t] = self.precisions.C.correspondance_matrix[precision_coord]
            else :
                # We have a parent, but it does not define our precisions
                print("Parent does not define precisions on C matrix")


            if(self.precisions.D.inherited) :
                # The precision is given by the upper states
                raise NotImplementedError("Precision on D matrices not implemented")
                for f in range(Nf):
                    if (self.precisions.D.BETA[f,t]<0):
                        precision_coord = (f,) + coordinates
                        self.precisions.D.BETA[f,t] = self.precisions.D.correspondance_matrix[precision_coord]
            else :
                # We have a parent, but it does not define our precisions
                print("Parent does not define precisions on D matrix")


            if(self.precisions.E.inherited) :
                # The precision is given by the upper states
                raise NotImplementedError("Precision on E matrices not implemented")
                if (self.precisions.policy.BETA[0,t]<0):
                    precision_coord = (0,) + coordinates
                    self.precisions.E.BETA[0,t] = self.precisions.E.correspondance_matrix[precision_coord]
            else :
                # We have a parent, but it does not define our precisions
                print("Parent does not define precisions on E matrix")
        self.precisions.fill_all_empty_BETAs(t,1)
        # self.precisions.fill_all_empty_BETAs(t,1.0)

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
                
                D_bar = precision_weight(self.D,self.precisions.D.BETA[:,t])
                B_bar = precision_weight(self.B,self.precisions.B.BETA[:,t])
                #print(D_bar,B_bar)
                if (t==0) :
                    prob_state = self.D[f]
                    #prob_state = D_bar[f]
                else :
                    prob_state = self.B[f][:,self.s[f,t-1],self.u[f,t-1]]
                    #prob_state = B_bar[f][:,self.s[f,t-1],self.u[f,t-1]]
                self.s[f,t] = np.argwhere(r.random() <= np.cumsum(prob_state,axis=0))[0]
        self.s[:,t] = self.s[:,t].astype(np.int)                            

        #True outcome definition
        for modality in range(Nmod):
            if (self.o[modality,t] < 0):
                # TODO : Outcome can be generated by an independent (parrallel) model
                # Or sample from likelihood given the hidden state
                ind = (slice(None),) + tuple(self.s[:,t])  # Indice corresponding to the current active states
                po = self.A[modality][ind]
                self.o[modality,t] = np.argwhere(r.random() <= np.cumsum(po,axis=0))[0]
        self.o[:,t] = self.o[:,t].astype(np.int)
        
        #TODO : outcome can be generated by a subordinate MDP (hierarchical model)

        #TODO : outcome can be generated by a variational filter

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

    def action_driven_loop(self):
        # Called after the GT values generation (generative process)
        Nmod = self.Nmod
        No = self.No

        Nf = self.Nf
        Ns = self.Ns 

        Np = self.Np
        Nu = self.Nu

        Ni = self.options.Ni
        t = self.t
        T = self.T

        N = self.options.T_horizon

        t0 = time.time()

        # Prior over subsequent states
        if (t>0):
            self.Q[t] = np.dot(self.b_kron[self.K[t-1]],self.Q[t-1])
        else :
            self.Q[t] = spm_kron(self.d)

        reduced_O = []
        for mod in range(Nmod):
            reduced_O.append(self.O[mod][:,t])

        # origin_node = tree_node(time=-1) # Initial belief about states
        # origin_node.states = np.copy(self.Q[0])
        # origin_node.action_density = 1.0
        # # Nodes for timesteps that actually happened
        # current_node = origin_node
        # for ti in range(t):
        #     current_node.add_child(self.Q[ti],1.0)
        #     current_node.children_nodes[0].data = self.u[:,ti]
        #     current_node = current_node.children_nodes[0]
        # # current_node contains the last state estimation (at t - 1)
        # #G,self.Q  = spm_forwards(reduced_O,self.Q,self.U,self.a,self.b_kron,self.c,self.e,self.a_ambiguity,self.a_complexity,t,T,min(T,t+N),current_node,t0=t,verbose=self.verbose)
        # tree = state_tree(origin_node)
        # origin_node.compute_subsequent_state_prob()     
        
        # The maximum horizon is the last timestep : T-1
        # Else, it is just the current time + temporal horizon :
        # If t horizon = 0, N = t, no loop
        # If t horizon = 1, N = t+1,there is one instance of t<N leading to another recursive tree search loop
        # If t horizon = 2, N = t+2, there are two nested instances of recursive search
        G,self.Q  = spm_forwards(reduced_O,self.Q,self.U,self.a,self.b_kron,self.c,self.e,self.a_ambiguity,self.a_complexity,self.b_complexity,t,T,min(T-1,t+N),t0=t,verbose=self.verbose)
        
        

        for i in range(t+1):
            qx = self.Q[i] # marginal posterior over hidden states
            # print(qx,spm_margin(qx,0))
            de_compressed = spm_dekron(qx,tuple(Ns))
            for factor in range(Nf):
                self.S[factor][:,i,t] = de_compressed[factor]

        

        for factor in range(Nf):
            for policy in range(Np):
                self.x[factor][:,t,policy] = self.S[factor][:,t,t]
                # TODO : policy dependent state ?

        for factor in range(Nf):
            self.X[factor][:,t] = self.S[factor][:,t,t]

        # posterior_over_policy & precision
        self.u_posterior[:,t] = softmax(G)
        self.precisions.policy.beta[0,t] = np.inner(self.u_posterior[:,t],nat_log(self.u_posterior[:,t])) # Entropy of posterior over policy distribution
        
        w = np.inner(self.u_posterior[:,t],nat_log(self.u_posterior[:,t])) 
        self.w[t] = w # Policy precision

        # Action selection :
        if (t<T-1):
            Ru = softmax(self.parameters.alpha * nat_log(self.u_posterior[:,t]))
            randomfloat = r.random()
            # print(self.u_posterior)
            # print(Ru)
            #print(np.cumsum(action_posterior_intermediate,axis=factor))
            ind = np.argwhere(randomfloat <= np.cumsum(Ru,axis=0))[0]
            #ind est de taille n où n est le nombre de transitions à faire :
            self.K[t] = ind[0]

            # Using action sequence free energies, we then pick the action independantly
            Pu = np.zeros((tuple(Nu)))
            for action_sequence in range(Np):
                sub = self.U[action_sequence,:]
                action_coordinate = tuple(sub)
                Pu[action_coordinate] = Pu[action_coordinate] + Ru[action_sequence]
            self.u[:,t][self.u[:,t]<0] = self.U[self.K[t],:]
            #print(b, np.cumsum(action_posterior_intermediate,axis=1),ind[0]) 
        return time.time() - t0

    def real_time_learn(self,ratio=0.5):
        """ Custom function to allow the agent to learn 'on the fly' at a reduced learning rate small eta seta = eta*reduction_factor"""
        if (self.options.learn_during_experience):
            learn_during_experience(self,ratio)
            #learn_during_experience(self,ratio)

    def tick(self):
        """ The layer gathers all data from parent & children states and performs inference"""
        t = self.t
        T = self.T 

        if(t<T):
            # print(np.round(self.x[0],2))
            self.get_ground_truth_values()
            # print("------------------")
            # print(str(t+1) + " / " + str(T))
            # print("------------------")
            if(self.policy_method == Policy_method.POLICY):
                #print("Policy driven loop engaged")
                state_inference_time = self.perform_state_inference()          
                policy_inference_time = self.perform_policy_inference()
                action_selection_time = self.perform_action_selection()
                self.rt[t] = state_inference_time + policy_inference_time
                self.t = self.t + 1
            elif (self.policy_method == Policy_method.ACTION):
                #print("Action driven loop engaged")
                total_time= self.action_driven_loop()
                self.rt[t] = total_time
                
                self.real_time_learn()

                self.t = self.t + 1
        else :
            print("No actions were conducted this round (time exceeded trial horizon)")


    def postrun(self,learn_aft = True,generate_electrophi_responses=False) :
        if(learn_aft):
            learn_from_experience(self,mem_dec_type=self.options.memory_decay,t05=self.options.decay_half_time)
        if(generate_electrophi_responses):
            generate_electroph_responses(self)
        #learn_from_experience(self,mem_dec_type=MemoryDecayType.STATIC)
        #learn_from_experience(self,mem_dec_type=MemoryDecayType.PROPORTIONAL)

    def run(self,learn = True):
        run_comps = []
        self.prep_trial()
        for t in range(self.T):
            self.tick()
        self.postrun(learn_aft = learn)
        run_comps.append(self.return_run_components())
        return run_comps
    
    def run_generator(self,learn = True):
        self.prep_trial()
        for t in range(self.T):
            self.tick()
            if (t<self.T-1):
                yield (self.return_run_components())
        self.postrun(learn_aft=learn)
        yield (self.return_run_components())
            

    def return_run_components(self):
        return_container = SimpleNamespace()

        # Results
        return_container.Q = flexible_copy(self.Q)
        return_container.o = flexible_copy(self.o)
        
        return_container.s = flexible_copy(self.s)    
        # Matrices
        return_container.a_ = flexible_copy(self.a_)
        return_container.A_ = flexible_copy(self.A_)

        return_container.b_ = flexible_copy(self.b_)
        return_container.B_ = flexible_copy(self.B_)

        return_container.c_ = flexible_copy(self.c_)
        return_container.C_ = flexible_copy(self.C_)

        return_container.d_ = flexible_copy(self.d_)
        return_container.D_ = flexible_copy(self.D_)

        return_container.e_ = flexible_copy(self.e_)
        return_container.E_ = flexible_copy(self.E_)

        return_container.rt = flexible_copy(self.rt)

        return return_container

    def return_void_state_space(self,populate = None):
        if (isNone(self.Ns)):
            return None
        else :
            shap = tuple(self.Ns)
            arr =np.zeros(shap)
            if (populate==None):
                return arr
            else :
                arr[populate] = 1.0
                return arr




if __name__ == "__main__":
    print("Hello there")