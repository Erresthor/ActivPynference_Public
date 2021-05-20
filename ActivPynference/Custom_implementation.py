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
r.seed() #For controlled variability

import numpy as np
import matplotlib.pyplot as plt
from function_toolbox import normalize,softmax,nat_log
from function_toolbox import spm_wnorm,cell_md_dot,md_dot, spm_cross,spm_KL_dir,spm_psi
from function_toolbox import G_epistemic_value

class MDP :
    def __init__(self):
        self.age = 0    #How many full iteration (experience + learning) has this MDP lived ?        
        self.name = ''
        self.savepath = ''
        self.packed = False
        
        # Simulation parameters
        self.T = 0
        self.Ni = 0
        
        
        #Model building blocks
        self.a_ = None
        self.b_ = None
        self.c_ = None
        self.d_ = None
        self.e_ = None
        
        self.A_ = None
        self.B_ = None
        self.C_ = None
        self.D_ = None
        self.E_ = None
        self.V_ = None
        
        
        self.s_ = None
        self.o_ = None
        
        #Modulation Parameters
        self.alpha = 512 # action precision
        self.beta = 1    # policy precision
        self.zeta = 3    # Occam window policies
        self.eta = 1     # learning rate
        self.tau = 4     # update time constant
        self.chi = 1/64  # Occam window updates
        self.erp = 64    # update reset
        self.time_constant = 4    #Constant for gradient descent
        
        # Variables
        self.Np = 0    #Number of policies
        
        self.No = []   #Number of outcomes for each MODALITY
        self.Nmod = len(self.No) #Number of modalities
        self.Ns = []   #Number of states for each FACTOR
        self.Nf = len(self.Ns) #Number of factors
        self.Nu = 0     # Number of controllable transitions
        
        
        #Output variables
        self.Fa = []            # Negative free energy of a
        self.Fb = []            # Negative free energy of b
        self.Fd = []            # Negative free energy of c
        self.Fe = []            # Negative free energy of d
        
        
        self.O = []             #Outcomes (expanded writing)
        self.o = None     #Outcomes (compressed writing)
        
        self.P = []             # Action posterior
        self.R = []             # Policy posterior
        self.u = None     # Actions
        
        self.Q = []             # State posteriors
        self.X = []             # BMA States
        self.s = None     # True states
        
        
        self.G = None     # EFE values
        self.F = None     # VFE values
        
        self.vn = []            # Neuronal prediction error 
        self.xn = []            # Neuronal encoding of hidden states
        self.un = []            # Neuronal encoding of policies
        self.wn = None    # Neuronal encoding of policy precision
        self.dn = None   # Simulated dopamine response
        
        
        #Archives for various runs
        
    
    def pack_model(self):
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

        #TODO : Initialize output variables ?

        self.packed = True
        print("Model packed.")
        
    
    def run(self,verbose=0):
        if (not(self.packed)) :
            self.pack_model()
        
        #Simulation variables
        T = self.T
        Ni = self.Ni
        
        Nf = self.Nf
        Ns = self.Ns
        
        Nmod = self.Nmod
        No = self.No
        
        Np = self.Np
        Nu = self.Nu
        
        TimeConst = self.time_constant
        
        

        # Likelihood model a / A
        if(self.a_):
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
        if(self.b_): # If we are learning the transition matrices
            b = normalize(self.b_)
            
            b_prior = []
            b_complexity = []
            for factor in range(Nf):   # For all factors
                b_prior.append(np.copy(self.b_[f]))
                b_complexity.append(spm_wnorm(b_prior[factor])*(b_prior[factor]>0))
        else :
            b = normalize(self.B_)
        B = normalize(self.B_)
        
        
        if(self.d_):
            d = normalize(self.d_)
            
            d_prior = []
            d_complexity = []
            for f in range(Nf):
                d_prior.append(np.copy(self.d_[f]))
                d_complexity.append(spm_wnorm(d_prior[f]))
        elif (self.D_) :
            d = normalize(self.D_)
        else :
            d = []
            for f in range(Nf):
                d.append(normalize(np.ones(Ns[f],)))
            self.D_ = d
        D = normalize(self.D_)
        
        
        if(self.e_.any()):
            E = self.e_
            e_prior = np.copy(self.e_)
        elif (self.E_):
            E = self.E_
        else :
            E = np.ones((Np,1))
        E = E/sum(E)
        
        
        C = []
        if (self.c_):
            c_prior = []
            for modality in range(Nmod):
                C.append(spm_psi(self.c_[modality] + 1./32))
                c_prior.append(np.copy(self.c_[modality]))      
        elif(self.C_):
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
            
            C[modality] = nat_log(softmax(C[modality]))
        
        
        V = self.V_
        
         
        #OUTCOMES
        outcomes = np.zeros((Nmod,T))
        O = []
        for modality in range(Nmod):
            O.append(np.zeros((No[modality],T)))
        true_states = np.zeros((Nf,T)).astype(np.int)
        # the value inside true states for a given factor can be :
        # 0 if not yet defined
        # x > 0 for a given state , with x in [1,Ns[f]]
        
        
        # POSTERIOR EXPECTATIONS OF HIDDEN STATES
        state_posterior = []
        normalized_firing_rates = []
        prediction_error = []
        for f in range(Nf) :
            q = np.ones((Ns[f],T,Np))/(Ns[f])
            state_posterior.append(q)
            
            nfs = np.zeros((Ni,Ns[f],T,T,Np))
            normalized_firing_rates.append(nfs)
            
            pe = np.zeros((Ni,Ns[f],T,T,Np))
            prediction_error.append(pe)
            
        BMA_states = []
        for f in range(Nf):
            BMA_states.append(np.zeros((Ns[f],T)))
        
        #Posterior over policies and action
        chosen_action = np.zeros((Nf,T-1)).astype(np.int)
        for f in range(Nf):
            if (Nu[f] == 1):
                chosen_action[f,:] = np.ones((T-1,))
       
                
        action_posterior = np.zeros(tuple(Nu)+(T,))
        policy_posteriors = np.ones((Np,T))/Np
        policy_posterior= np.zeros((Np,T))
        policy_posterior_updates = np.zeros((Np,T*Ni))
        policy_prior = np.zeros((Np,T))
        
        #Variable initializations for forthcoming loop
        F = np.zeros((Np,T))
        G = np.zeros((Np,T))
        posterior_beta = 1
        gamma = [1/posterior_beta] #Expected Free Energy precision
        gamma_update = np.zeros((T*Ni,))
        
        
        # Posterior expectations of hidden states
        xn = []
        vn = []
        x = []
        X = []
        for f in range(Nf):        
            xn.append(np.zeros((Ni,Ns[f],1,1,Np)) + 1./Ns[f])
            vn.append(np.zeros((Ni,Ns[f],1,1,Np)))
            x.append(np.zeros((Ns[f],T,Np)) + 1./Ns[f])
            X.append(np.tile(D[f],(1,1)))
            for k in range(Np):
                x[f][:,0,k] = D[f]
        #Posterior over policies and action
        P = np.zeros(tuple(Nu))
        un = np.zeros((Np,))
        u = np.zeros((Np,))                 # = chosen action
        if (Np == 1) :
            u = np.ones((Np,T))
        
        #states
        s = np.full((Nf,T),None)  # = true states
        if (self.s_):
            s = np.copy(self.s_)
        self.s_ = np.copy(s)
        #s = s.astype(np.int)
        
        #outcomes
        o = np.full((Nmod,T),None)
        if (self.o_):
            o = np.copy(self.o_)
        self.o_ = np.copy(o)
        #o = o.astype(np.int)
        
        p = range(Np) # Indices of allowable policies
        
        if (verbose):
            print()
            print("EXPERIENCE :")
        
        
             
        for t in range(T) :
            if(verbose):
                print("----------------------------")
                print(t+1)
                print("----------------------------")
        #    print()
        #    print()
            Ft = np.zeros((T,Ni,t+1,Nf))
            
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
                if (true_states[f,t] == None) :
                    if (t==0) :
                        prob_state = D[f]
                    else :
                        #print(B[f][:,true_states[f,t-1],chosen_action[f,t-1]])
                        prob_state = B[f][:,true_states[f,t-1],chosen_action[f,t-1]]
                    true_states[f,t] = np.argwhere(r.random() <= np.cumsum(prob_state,axis=0))[0]
            true_states = true_states.astype(np.int)
            
            #Posterior predictive density over hidden external states
            xq = []
            xqq = []
            print(X[0])
            for f in range(Nf) :
                if (t ==0) :
                    xqq.append(X[f][:,t])
                else :
                    print(b[factor][:,:,chosen_action[f,t-1]].shape)
                    print(X[f][:,t-1].shape)
                    xqq.append(np.dot(np.squeeze(b[factor][:,:,chosen_action[f,t-1]]),X[f][:,t-1]))
            
            print(xqq)
            
            
            
            
            
            
            #Sample observations
            for modality in range(Nmod):
                #print(A[1][true_states[0,t],true_states[1,t]])
                # TODO : imo, the expression below should be : outcomes[modality,t] = np.argwhere(r.random() <= np.cumsum(A[modality][:,true_states[0,t],true_states[1,t]],axis=0))[0]
                # Note the different observation likelihood matrix used
                # After some experiments, it seems to be the best expression wrt outcomes[modality,t] = np.argwhere(r.random() <= np.cumsum(a[modality][:,true_states[0,t],true_states[1,t]],axis=0))[0]
                outcomes[modality,t] = np.argwhere(r.random() <= np.cumsum(A[modality][:,true_states[0,t],true_states[1,t]],axis=0))[0]
            outcomes = outcomes.astype(np.int)
            
            
            for modality in range(Nmod):
                vec = np.zeros((1,No[modality]))
                #print(vec.shape)
                index = outcomes[modality,t]
                #print(index)
                vec[0,index] = 1
                O[modality][:,t] = vec
        
            if (verbose) :
                print()
                print("------------------------- Generative process at time t= " + str(t) + " ----------------------------")
                print("True states at t = " + str(t) +" || " + str(true_states[:,t]))
                print("Outcomes at t = " + str(t) +" || " + str(outcomes[:,t]))
                print() 
                
            # END OF GENERATIVE PROCESS !!
            #-------------------------------------------------------------------------------------------------
            
            
            #GENERATIVE MODEL !!
            #-------------------------------------------------------------------------------------------------
                print("-------------------------- Generative model at time t= " + str(t) + " -----------------------------")
            
            # marginal message passing (minimize F and infer posterior over states)
            #----------------------------------------------------------------------
            for policy in range(Np) :
                for iteration in range(Ni) :
                    for factor in range(Nf) :
                        lnAo = np.zeros(state_posterior[factor].shape[:-1])
                        for tau in range(T) :
                            v_depolarization = nat_log(state_posterior[factor][:,tau,policy]) # = log(s(tau,pi))
                            
                            if (tau <= t) : #If tau has already happened
                                for modality in range(Nmod):
                                    lnA =nat_log(a[modality][outcomes[modality,tau],:,:])
                                    for fj in range(Nf):
                                        if fj != factor :
                                            lnAs = md_dot(lnA,state_posterior[fj][:,tau,policy],fj)
                                            # TODO : choose the best between the above and 
                                            #lnAs = md_dot(lnA,state_posterior[fj][:,tau],fj)
                                            lnA = lnAs
                                    lnAo [:,tau] = lnAo[:,tau] + lnA
                                
                            if tau == 0 :
                                lnD = nat_log(d[factor])
                                lnBs = nat_log(np.dot(normalize(b[factor][:,:,V[tau,policy,factor]].T),state_posterior[factor][:,tau+1,policy]))
                            elif (tau==T-1) :
                                lnD = nat_log(np.dot(np.squeeze(b[factor][:,:,V[tau-1,policy,factor]]),state_posterior[factor][:,tau-1,policy]))
                                lnBs = np.zeros(d[factor].shape)
                            else :
                                
                                lnD = nat_log(np.dot(np.squeeze(b[factor][:,:,V[tau-1,policy,factor]]),state_posterior[factor][:,tau-1,policy]))
                                lnBs = nat_log(np.dot(normalize(b[factor][:,:,V[tau,policy,factor]].T),state_posterior[factor][:,tau+1,policy]))
                                
                            sum_of_messages_at_node_s = 0.5*lnD + 0.5*lnBs + lnAo[:,tau]
                            v_depolarization = v_depolarization + (sum_of_messages_at_node_s - v_depolarization)/TimeConst
                            
                            
                            Ft[tau,iteration,t,factor] = np.dot(state_posterior[factor][:,tau,policy].T,0.5*lnD + 0.5*lnBs - lnAo[:,tau] - nat_log(state_posterior[factor][:,tau,policy]))
                            state_posterior[factor][:,tau,policy] = softmax(v_depolarization)
                            
                            normalized_firing_rates[factor][iteration,:,tau,t,policy] = state_posterior[factor][:,tau,policy]
                            prediction_error[factor][iteration,:,tau,t,policy] = v_depolarization
                        #End loop on tau
                    #End loop on factor
                #End loop on iteration
                
                #Now, for each policy, we calculate de VFE :
                Fintermediate = np.sum(Ft,axis=-1) #Sum over factors
                #print(Fintermediate.shape) #--> tau x iteration x t
                Fintermediate = np.sum(Fintermediate,axis=0) #Sum over taus
                F[policy,t] = Fintermediate[-1,-1]
            #End loop on policy   
            
            if(verbose>2):
                print("State posteriors at t = " +str(t) + "  For factor Behavioural state :")
                for i in range((state_posterior[1].shape[2])) :
                    disp = "> Policy " + str(i) + " ----------------------- " 
                    print(disp)
                    print(str(np.round(state_posterior[1][:,:,i],3)))
                print()
            
            if (verbose):
                print("VFE at t = " +str(t) + "   :")
                print(np.round(F[:,t] ,3))
                print()




            # ---------------------------------------------------------------------------------
            #Calculus of Expected Free Enegry (G) under each policy
            Gintermediate = np.zeros((Np,))
            horizon = T
            if(verbose>1) :
                print("States expectations according to policy at time t = " + str(t))
                    
            
            for policy in range(Np) :                
                if (self.d_):
                    for factor in range (Nf):
                        Gintermediate[policy] = Gintermediate[policy] - np.inner(d_complexity[factor].T,state_posterior[factor][:,0,policy])
                        
        
                for timestep in range(t,horizon):
                    Expected_states = []
                    for factor in range(Nf):
                        Expected_states.append(state_posterior[factor][:,timestep,policy])


                    if(verbose>1):
                        print("     --> At timestep = " + str(timestep) +" >= t")
                        print("          > Context states :     "+ str(np.round(Expected_states[0],2)))
                        print("          > Behavioural states : "+ str(np.round(Expected_states[1],2)))
                    

                    Gintermediate[policy] = Gintermediate[policy] + G_epistemic_value(a,Expected_states)
                    

                    for modality in range(Nmod):

                        predictive_observation_posterior = cell_md_dot(a[modality],Expected_states)                      
                        Gintermediate[policy] = Gintermediate[policy] + np.dot(predictive_observation_posterior.T,C[modality][:,timestep])
                        #TODO : choose between the above and 
                        # Gintermediate[policy] = Gintermediate[policy] + np.dot(predictive_observation_posterior.T,C[modality][:,timestep])
                        

                        if (self.a_) :
                            Gintermediate[policy] = Gintermediate[policy] - cell_md_dot(a_complexity[modality],[predictive_observation_posterior] + Expected_states[:])
                        
                                        
                        
                        if(verbose>1):
                            disp= "predictive_observation_posterior = " + str(np.round(predictive_observation_posterior,2))
                            disp  = disp + " Potential gain = " + str(np.round(C[modality][:,timestep],2))
                            print("         For policy " +str(policy) +" and modality " +str(modality) + "  | " +  disp)
                            print("                                      | bayesian suprise = " + str(np.round(-cell_md_dot(a_complexity[modality],[predictive_observation_posterior] + Expected_states[:]),2)))
                            print("                                      | Gintermediate = " + str(np.round(Gintermediate[policy],2))        )
                    
            G[:,t] = Gintermediate
            
            if(verbose) :
                #End of the loop on policies
                print()
                print("EFE at t = " +str(t) + "   :")
                print(np.round(G[:,t],3))
                print()

            # Policy inference ,precision updates, etc.
            if (t>0) :
                gamma.append(gamma[t-1])
            for iteration in range(Ni):
                policy_prior[:,t] = softmax(nat_log(E) + gamma[t]*G[:,t])
                
                policy_posteriors[:,t] = softmax(nat_log(E) + gamma[t]*G[:,t] + F[:,t])
                
                
                
                #print(G)
                #EFE precision (beta)
                beta_update = np.dot((policy_posteriors[:,t] - policy_prior[:,t]).T, G[:,t])
                #print(beta_update)
                dFd_gamma = posterior_beta - self.beta + beta_update #Free Energy gradient wrt gamma
                posterior_beta = posterior_beta-dFd_gamma/2
                gamma[t] = 1 / posterior_beta
                
                #Dopamine responses 
                n = (t)*Ni + iteration
                gamma_update[n] = (gamma[t]) #gamma_update[n,1] = gamma[t]
                policy_posterior_updates[:,n] = policy_posteriors[:,t]
                policy_posterior[:,t] = policy_posteriors[:,t]
                
            if(verbose>2):
                print("Policy prior at t = " +str(t) + "   :")
                print(np.round(policy_prior[:,t],3))
                print("Policy posterior at t = " +str(t) + "   :")
                print(np.round(policy_posterior[:,t],3))
                print()
            
            #Bayesian model average of hidden states
            for factor in range(Nf) :
                for tau in range(T):
                    BMA_states[factor][:,tau] = np.dot(state_posterior[factor][:,tau,:],policy_posteriors[:,t])
                    
            # ---------------------------------------------------------------------------------
            #Action selection
        #    % The probability of emitting each particular action is a softmax function 
        #    % of a vector containing the probability of each action summed over 
        #    % each policy. E.g. if there are three policies, a posterior over policies of 
        #    % [.4 .4 .2], and two possible actions, with policy 1 and 2 leading 
        #    % to action 1, and policy 3 leading to action 2, the probability of 
        #    % each action is [.8 .2]. This vector is then passed through a softmax function 
        #    % controlled by the inverse temperature parameter alpha which by default is extremely 
        #    % large (alpha = 512), leading to deterministic selection of the action with 
        #    % the highest probability. 
        #    print(policy_prior)
        #    print(policy_posterior)
            if (t<T-1):
                #Marginal posterior over actions
                action_posterior_intermediate = np.zeros(tuple(Nu))
                for policy in range(Np):
                    sub = V[t,policy,:] # coordinates for the corresponding action wrt t and policy
                    action_coordinate = tuple(sub)
                    action_posterior_intermediate[action_coordinate] = action_posterior_intermediate[action_coordinate] + policy_posteriors[policy,t]

                action_posterior_intermediate = softmax(self.alpha*nat_log(action_posterior_intermediate))
                action_posterior[...,t] = action_posterior_intermediate
                # Next action : sampled from marginal posterior
                
                for factor in range(Nf):
                    if(Nu[factor]>1) :
                        randomfloat = r.random()
                        #print(np.cumsum(action_posterior_intermediate,axis=factor))
                        ind = np.argwhere(randomfloat <= np.cumsum(action_posterior_intermediate,axis=factor))[0]
                        #ind est de taille n où n est le nombre de transitions à faire :
                        chosen_action[factor,t] = ind[factor]
                        #print(b, np.cumsum(action_posterior_intermediate,axis=1),ind[0])
            #print(ControlIndex)
            if(verbose):
                print("-------------------------------------------------------------------------------------")
                print("*************************************************************************************")
        
        #    print("******************************")
        #    print("BMA STATES at time " + str(t) + " : ")
        #    print(np.round(BMA_states[0],3))
        #    print(np.round(BMA_states[1],3))
        #    print()
        #    print("******************************")
        if(verbose):
            print("Loop over time points ended without issues.")
            print()
               
        #-----------------------------------------------------------------------------------------------
        #Learning !              
        
            print("LEARNING:")
        
        if (self.a_):
            for t in range(T):
                for modality in range(Nmod):
                    #print(t,modality)
                    a_learning = O[modality][:,t]
                    for factor in range(Nf):
                        a_learning = spm_cross(a_learning,BMA_states[factor][:,t])
                    a_learning = a_learning*(self.a_[modality]>0)
                    self.a_[modality] = self.a_[modality] + a_learning*self.eta
        
        
        if(self.d_):
            for factor in range(Nf):
                self.d_[factor] = self.d_[factor] +  (self.eta*BMA_states[factor][:,0])*(self.d_[factor]>0)
        
        if(self.e_.any()) :
            #print(policy_posterior[:,T-1],e_)
            self.e_ = self.e_ + self.eta*policy_posterior[:,T-1]
            
        
        #Free energy of concentration parameters
        self.Fa = []
        if(self.a_):
            for modality in range(Nmod):
                self.Fa.append(-spm_KL_dir(self.a_[modality],a_prior[modality]))
        
        self.Fd = []
        if (self.d_):
            for factor in range(Nf):
                self.Fd.append(-spm_KL_dir(self.d_[factor],d_prior[factor]))
        
        self.Fe = []
        if (self.e_.any()) :
            self.Fe.append(-spm_KL_dir(self.e_,E))
            #Fe.append(-spm_KL_dir(self.e_,e_prior))
            #TODO if it is compared to its prior (else it will likely only grow due to difference between normalized and not normalized variables)
        
        
        
        phasic_dopamine = []
        if (Np>1):
            constante_dop = 8.0 
            phasic_dopamine = constante_dop*np.gradient(gamma_update) + gamma_update/constante_dop
        
        if (verbose):
            print("Learning complete.")
            print()
            
        BMA_norm_firing_rates = []
        BMA_prediction_error = []
        for factor in range(Nf):
            BMA_norm_firing_rates.append(np.zeros((Ni,Ns[factor],T,T)))
            BMA_prediction_error.append(np.zeros((Ni,Ns[factor],T,T)))
            for t in range(T):
                for policy in range(Np):
                    BMA_norm_firing_rates[factor][:,:,:,t] = BMA_norm_firing_rates[factor][:,:,:,t] + policy_posterior[policy,t]*normalized_firing_rates[factor][:,:,:,t,policy]
                    BMA_prediction_error[factor][:,:,:,t] = BMA_prediction_error[factor][:,:,:,t] + policy_posterior[policy,t]*prediction_error[factor][:,:,:,t,policy]
        
        #print(np.round(BMA_prediction_error[0][-1,:,:,:],2))
        
        
        self.P = action_posterior
        self.R = policy_posterior
        self.Q = state_posterior
        self.X = BMA_states
        self.G = G
        self.F = F
        self.O = O
        self.o = outcomes
        
        
        self.w = gamma
        self.vn = BMA_prediction_error
        self.xn = BMA_norm_firing_rates
        self.un = policy_posterior_updates
        self.wn = gamma_update
        self.dn = phasic_dopamine
        
        self.age = self.age + 1
        



model = MDP()
     
# SET UP MODEL STRUCTURE ------------------------------------------------------
print("-------------------------------------------------------------")
print("------------------SETTING UP MODEL STRUCTURE-----------------")
print("-------------------------------------------------------------")

#Points within a trial
model.T = 3 
T = model.T
model.Ni = 16

# Priors about initial states
print("\n Priors about initial states D & d")
# Prior probabilities about initial states in the generative process
D_ =[]
# Context state factor
D_.append(np.array([1,0])) #[Left better, right better]
# Behaviour state factor
D_.append(np.array([1,0,0,0])) #{'start','hint','choose-left','choose-right'}
model.D_ = D_
print("D : " + str(D_))

# Prior beliefs about initial states in the generative process
d_ =[]
# Context beliefs
d_.append(np.array([0.25,0.25])) #[Left better, right better]
# Behaviour beliefs
d_.append(np.array([1,0,0,0])) #{'start','hint','choose-left','choose-right'}
model.d_ = d_
print("d : " + str(d_))
print("-------------------------------------")

# State Outcome mapping and beliefs
print(" State Outcome mapping and beliefs A & a")
# Prior probabilities about initial states in the generative process
Ns = [D_[0].shape[0],D_[1].shape[0]] #(Number of states)
A_ = []
#Mapping from states to observed hints, accross behaviour states (non represented)
#
# [ .  . ]  No hint
# [ .  . ]  Machine Left Hint            Rows = observations
# [ .  . ]  Machine Right Hint
# Left Right
# Columns = context state
A_obs_hints = np.zeros((3,Ns[0],Ns[1]))
A_obs_hints[0,:,:] = 1
pHA = 1
A_obs_hints[:,:,1] = np.array([[0,0],
                         [pHA, 1-pHA],
                         [1-pHA,pHA]]) # Behaviour ste "hint" gives an observed hint
print(A_obs_hints.shape)
    
    
#Mapping from states to outcome (win / loss / null), accross behaviour states (non represented)
#
# [ .  . ]  Null
# [ .  . ]  Win           Rows = observations
# [ .  . ]  Loss
#
# Columns = context state
A_obs_outcome = np.zeros((3,Ns[0],Ns[1]))
A_obs_outcome[0,:,0:2] = 1
pWin = 1
A_obs_outcome[:,:,2] = np.array([[0,0],   # If we choose left, what is the probability of achieving win / loss 
                         [pWin, 1-pWin],
                         [1-pWin,pWin]]) # Choice gives an observable outcome
               # If true = left, right
A_obs_outcome[:,:,3] = np.array([[0,0],     # If we choose right, what is the probability of achieving win / loss 
                         [1-pWin, pWin],
                         [pWin,1-pWin]]) # Choice gives an observable outcome
              # If true = left, right
print(A_obs_outcome.shape)

#Mapping from behaviour states to observed behaviour
#
# [ .  .  .  .] start
# [ .  .  .  .] hint
# [ .  .  .  .] choose left         Row = Behaviour state
# [ .  .  .  .] choose right
#  s   h  l  r
#
# 3rd dimension = observed behaviour
# The 2nd dimension maps the dependance on context state
A_obs_behaviour = np.zeros((Ns[1],Ns[0],Ns[1]))
for i in range (Ns[1]) :
    A_obs_behaviour[i,:,i] = np.array([1,1])

print(A_obs_behaviour.shape)

A_ = [A_obs_hints,A_obs_outcome,A_obs_behaviour]
model.A_ = A_
print(" A : " + str(A_))
print("-------------------------------------")

# Transition matrixes between hidden states ( = control states)
print("Transition between states : B")
B_ = []
#a. Transition between context states --> The agent cannot act so there is only one :
B_context_states = np.array([[[1],[0]],
                             [[0],[1]]])
B_.append(B_context_states)
#b. Transition between behavioural states --> 4 actions
B_behav_states = np.zeros((Ns[1],Ns[1],Ns[1]))
# - 0 --> Move to start from any state
B_behav_states[0,:,0] = 1
# - 1 --> Move to hint from any state
B_behav_states[1,:,1] = 1
# - 2 --> Move to choose left from any state
B_behav_states[2,:,2] = 1
# - 3 --> Move to choose right from any state
B_behav_states[3,:,3] = 1
B_.append(B_behav_states)
print(" B : " + str(B_))
print(B_[0].shape, B_[1].shape)
model.B_  = B_
print("-------------------------------------")

# Preferred outcomes
print(" Preferred outcomes : C")
# One matrix per outcome modality. Each row is an observation, and each
# columns is a time point. Negative values indicate lower preference,
# positive values indicate a high preference. Stronger preferences promote
# risky choices and reduced information-seeking.
No = [A_[0].shape[0],A_[1].shape[0],A_[2].shape[0]]



la = 1 #Loss aversion
rs = 3 #reward seeking

C_hints = np.zeros((No[0],T))
C_win_loss = np.zeros((No[1],T))
C_win_loss = np.array([[0,0,0],     #null
                       [0,rs,rs/2],  #win
                       [0,-la,-la]]) #loss
C_observed_behaviour = np.zeros((No[2],T))
C_ = [C_hints,C_win_loss,C_observed_behaviour]
print("C : " + str(C_))
model.C_ = C_
print("-------------------------------------")


# Policies
print(" Allowable policies : U / V")
Np = 5 #Number of policies
Nf = 2 #Number of state factors
V_ = np.zeros((T-1,Np,Nf))
V_[:,:,0]= np.array([[0,0,0,0,0],      # T = 2
                     [0,0,0,0,0]])     # T = 3  row = time point
    #                colums = possible course of action in this modality (0 -->context states)
V_[:,:,1] = np.array([[0,1,1,2,3],      # T = 2
                     [0,2,3,0,0]])     # T = 3  row = time point in this modality (1 -->behavioural states)
    #                colums = possible course of action
print(V_, V_.shape)
V_ = V_.astype(np.int)
model.V_ = V_
print("-------------------------------------")

#Habits
print(" Habits : E & e")
E_ = None
model.E_ = E_
print(E_)
print("-------------------------------------")

#Other parameters
print(" Parameters")
model.eta = 1 #Learning rate
model.beta = 1 # expected precision in EFE(pi), higher beta --> lower expected precision
         # low beta --> high influence of habits, less deterministic policiy selection
model.alpha = 32 # Inverse temperature / Action precision
            # How much randomness in selecting actions
            # (high alpha --> more deterministic)
model.erp = 1  # degree of belief resetter at each time point
         # ( 1 means no reset bet. time points, higher values mean more loss in confidence)
model.tau = 12 #Time constant for evidence accumulation
            # magnitude of updates at each iteration of gradient descent
            # high tau --> smaller updates, smaller convergence, greater  stability
print("eta = " + str(model.eta) + "\nbeta = " + str(model.beta) + "\nalpha = " + str(model.alpha) + "\nerp = " + str(model.erp) + "\ntau = " + str(model.tau))
print("-------------------------------------")



a_ = []
for mod in range (len(A_)):
    a_.append(np.copy(A_[mod])*200)

a_[0][:,:,1] = np.array([[0,0],
                        [0.25,0.25],
                        [0.25,0.25]])
model.a_ = a_
    

e_ = np.ones((Np,))
model.e_ = e_
    


model.pack_model()

N = 100
for i in range(N):
    print(str(int(100*i/N)) + str("  %"))
    model.run()
    if (i%10 < 1e-6) :
        dop = model.dn
        x= np.linspace(0,dop.shape[0],dop.shape[0])
        plt.plot(x,dop)

    
print(model.o)
print(model.a_[0])


dop = model.dn
x= np.linspace(0,dop.shape[0],dop.shape[0])
plt.plot(x,dop)
plt.show()












#
#
#
#
#print()
#print()
#print("----------------------------------------------------------------------")         
#print("----------------------------------------------------------------------")
#print("----------------------------    RESULTS   ----------------------------")
#print("----------------------------------------------------------------------")
#print("----------------------------------------------------------------------")
#



def display_results(O,true_states,pol_post,action_posterior):
    print()
    print("---Simulation Results for ActiveInference Tutorial---")
    print("-----------------------------------------------------")  
    print(" AGENT BEHAVIOUR : ")
    for obs in range (T):
        disp = "       Observation at t="+str(obs)+" : "
        a = O[2][:,obs]
        i = np.argwhere(a==1)[0][0]
        if(i==0):
            disp = disp + str(" --> Start        ")
        elif(i==1):
            disp = disp + str(" --> Hint foraging")
        elif(i==2) :
            disp = disp + str(" --> Choose Left  ")
        elif(i==3) :
            disp = disp + str(" --> Choose Right ")
        b = true_states[1,obs]
        disp = disp + str("      Real behavioural state :  --> ")
        if(b==0):
            disp = disp + ("Start")
        if (b==1) :
            disp = disp + ("Hint foraging")
        if (b==2) :
            disp = disp + ("Choose Left")
        if (b==3):
            disp = disp + ("Choose Right")
        print(disp)
    print("-----------------------------------------------------")        
    print(" HINTS : ")
    for obs in range (T):
        disp = "       Observation at t="+str(obs)+" : "
        a = O[0][:,obs]
        i = np.argwhere(a==1)[0][0]
        if(i==0):
            disp = disp + str(" --> No hint     ")
        elif(i==1):
            disp = disp + str(" --> Hint : Left ")
        elif(i==2) :
            disp = disp + str(" --> Hint : Right")
        disp = disp + str("       Real situation state :    --> ")
        b = true_states[0,obs]
        if(b==0):
            disp = disp + ("Left")
        if (b==1) :
            disp = disp + ("Right")
        print(disp)
    print("-----------------------------------------------------") 
    print(" OUTCOME : ")
    for obs in range (T):
        disp = "       Observation at t="+str(obs)+" : "
        a = O[1][:,obs]
        i = np.argwhere(a==1)[0][0]
        if(i==0):
            disp = disp + str(" --> No outcome")
        elif(i==1):
            disp = disp + str(" --> WIN           :-D")
        elif(i==2) :
            disp = disp + str(" --> LOSS          >:(")
        print(disp)
    print("-----------------------------------------------------")
    print(" POLICIES : ")
    for obs in range (T-1):
        disp = "       Most likely Policy(ies) infered at t="+str(obs)+ " and to pursue at t=" + str(obs + 1) + " : "
        k = np.argwhere(pol_post[:,obs] == np.max(pol_post[:,obs]))
        policiestochoose = ""
        for minik in range(len(k)) :
            policiestochoose = policiestochoose + "  " +  str(k[minik][0] + 1)
        disp = disp + policiestochoose + "   Policy posterior  :" + str(np.round(pol_post[:,obs],3))
        print(disp)
    print("-----------------------------------------------------")
    print(" ACTIONS : ")
    for obs in range (T-1):
        disp = "       Most likely Action infered at t="+str(obs)+" and to do at t= " + str(obs+1) + " : "
        k = np.argwhere(action_posterior[0,:,obs] == np.max(action_posterior[0,:,obs]))
        for b in k :
            bie = b[0]
            if(bie==0):
                disp = disp + (" Start ")
            if (bie==1) :
                disp = disp + (" Hint foraging ")
            if (bie==2) :
                disp = disp + (" Choose Left ")
            if (bie==3):
                disp = disp + (" Choose Right ")
        print(disp)
    print("-----------------------------------------------------")

#display_results(O,true_states,policy_posterior,action_posterior)