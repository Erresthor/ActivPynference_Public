
import time
import numpy as np
from ...base.miscellaneous_toolbox import isField
from ...base.function_toolbox import normalize,softmax
from ...base.function_toolbox import spm_kron,spm_margin,spm_dekron,spm_wnorm,nat_log,spm_psi

from .cache import layer_cache

def initialize_sizes(model_layer,cache):
    """This function initialize the sizes of all blocks"""
    T = model_layer.T

    Nf = len(model_layer.d)

    if (isField(model_layer.U)):
        Np = model_layer.U_.shape[0] # Number of allowable set of actions
    else :
        print("No Action space defined. Are you sure you want to perform a sophisticated inference scheme ?")
        return

    Nu = []    # Number of allowable actions for each factor
    for f in range(Nf) :
        assert model_layer.b[f].ndim > 2,"Agent transition model b["+str(f)+"] has too little dimensions ("+str(model_layer.b_[f].ndim)+")"
        Nu.append(model_layer.b[f].shape[2])

    Nmod = len(model_layer.a)
    No = []
    for i in range(Nmod) :
        assert model_layer.a[i].ndim > 1,"Agent perception model a["+str(i)+"] has too little dimensions ("+str(model_layer.a[i].ndim)+")"
        No.append(model_layer.a[i].shape[0])
    
    Ns = []
    for f in range(Nf):
        assert model_layer.d[f].ndim > 0,"Agent initial state model d["+str(i)+"] has too little dimensions ("+str(model_layer.d[f].ndim)+")"
        Ns.append(model_layer.d[i].shape[0])
    
    cache.dims.T = T

    cache.dims.Nf = Nf
    cache.dims.Ns = Ns
    
    cache.dims.Nmod = Nmod
    cache.dims.No = No
    
    cache.dims.Np = Np
    cache.dims.Nu = Nu

def initialize_model_vars(model_layer,cache):
    epsilon = 1e-10
    T = cache.dims.T
    Nmod = cache.dims.Nmod
    No = cache.dims.No
    Nf = cache.dims.Nf
    Ns = cache.dims.Ns  
    Np = cache.dims.Np
    Nu = cache.dims.Nu

    a = normalize(model_layer.a)    # <=> A{m,g}
    a_prior = []                        # <=> pA{m,g}
    a_novelty = []                  # <=> W{m,g}
    a_ambiguity = []                # <=> H{m,g}
    for modality in range(Nmod):
        a_prior.append(np.copy(model_layer.a))   # Prior dirichlet allocation
        a_novelty.append(spm_wnorm(a_prior[modality])*(a_prior[modality]>epsilon) )
        a_ambiguity.append(np.sum(a[modality]*nat_log(a[modality]),0))

    b = normalize(model_layer.b)
    b_prior = []
    b_complexity = []
    for factor in range(Nf):   # For all factors
        b_prior.append(np.copy(model_layer.b[factor])) # Prior dirichlet allocation
        b_complexity.append(spm_wnorm(b_prior[factor])*(b_prior[factor]>epsilon))

    # Kronecker form of policies :
    # Some way of "compressing" multiple factors into a single matrix 
    # Different from Matlab script, because our kronecker product orders dimension differently
    b_kron = [] 
    b_complex_kron = []
    for k in range(Np) :  # For each potential policy
        b_kron.append(1)
        b_complex_kron.append(1)
        for f in range(Nf):  # And each potential state factor
            #b_kron[k] = spm_kron(b[f][:,:,layer_input.U_[k,f]],b_kron[k])
            b_kron[k] = spm_kron(b_kron[k],b[f][:,:,model_layer.U[k,f]])
            b_complex_kron[k] = spm_kron(b_complex_kron[k],b_complexity[f][:,:,model_layer.U[k,f]])                
    
    # prior over initial states d/D
    d = normalize(model_layer.d)
    d_prior = []
    d_complexity = []
    for f in range(Nf):
        d_prior.append(np.copy(model_layer.d[f])) # Prior dirichlet allocation
        d_complexity.append(spm_wnorm(d_prior[f]))
    
    # Habit E
    e = normalize(model_layer.e)
    e_prior = np.copy(model_layer.e)
    e = nat_log(e)
    
    # Preferences C
    c = []
    c_prior = []
    for modality in range(Nmod):
        c.append(spm_psi(model_layer.c[modality] + 1./32))
        c_prior.append(np.copy(model_layer.c[modality])) 

    
    for modality in range(Nmod):
        assert(c[modality].ndim>1),"Preference matrix c should be at least a 2 dimensional matrix. If preferences are time-invariant, the second dimension should be of size 1."
        if (c[modality].shape[1] == 1) :
            c[modality] = np.tile(c[modality],(1,T))
            
            model_layer.c[modality] = np.tile(model_layer.c[modality],(1,T))
            
            c_prior[modality] = np.tile(c_prior[modality],(1,T))
        c[modality] = nat_log(softmax(C[modality],0))
    
    U = model_layer.U.astype(np.int)

    # --------------------------------------------------------
    # GENERATIVE MODEL COMPONENTS
    cache.vars.a = a
    cache.vars.a_prior = a_prior
    cache.vars.a_novelty = a_novelty
    cache.vars.a_ambiguity = a_ambiguity

    cache.vars.b = b
    cache.vars.b_prior = b_prior
    cache.vars.b_complexity = b_complexity
    cache.vars.b_kron = b_kron
    cache.vars.b_complexity = b_complex_kron
    
    cache.vars.c = c
    cache.vars.c_prior = c_prior
    
    cache.vars.d = d
    cache.vars.d_prior = d_prior
    cache.vars.d_complexity = d_complexity
    
    cache.vars.e = e
    cache.vars.e_prior = e

def prep_cache(model_layer,overwrite = False):
    if not(isField(model_layer.cache)) or overwrite :
        cache = layer_cache()
        initialize_sizes(model_layer,cache)
        initialize_model_vars(model_layer,cache)


def fix_values(o,u) :
    # --------------------------------------------------------

    # the value inside observations/ true states  can be:
    # <0 if not yet defined : it will be created by the generative process
    # x in [0,No[modality]-1]/[0,Ns[factor]-1] for a given modality/factor at a given time if we want to explicit some value

    #OUTCOMES -------------------------------------------------------------
    layer_input.O = []
    for modality in range(Nmod):
        layer_input.O.append(np.zeros((No[modality],T)))

    o = np.full((Nmod,T),-1)      
    if isField(layer_input.o_):  # There is an "o" matrix with fixed values ?
        if (type(layer_input.o_) == np.ndarray) :
            o[layer_input.o_ >=0] = layer_input.o_[layer_input.o_>=0]
            
        # If there are fixed values for the observations, they will be copied
    layer_input.all_outcomes_input = (np.sum(o<0)==0)  # If we have all outcomes as input, no need to infer them 
    layer_input.o = np.copy(o)

    #STATES ------------------------------------------------------------------
    # true states 
    s = np.full((Nf,T),-1)
    if isField(layer_input.s_):
        if (type(layer_input.s_) == np.ndarray) :
            s[layer_input.s_ >=0] = layer_input.s_[layer_input.s_ >=0]
    layer_input.all_states_input = (np.sum(s<0)==0)  # If we have all states as input, no need to infer them 
    layer_input.s = np.copy(s)
    
    # Posterior expectations of hidden states
    # state posteriors
    layer_input.x = []
    layer_input.xn = []
    layer_input.X = []
    layer_input.X_archive = []
    layer_input.S = []
    for f in range(Nf):
        layer_input.x.append(np.zeros((Ns[f],T,Np)) + 1./Ns[f])                     # Posterior expectation of all hidden states depending on each policy
        layer_input.xn.append(np.zeros((Ni,Ns[f],T,T,Np)) + 1./Ns[f])
        layer_input.X.append(np.tile(np.reshape(d[f],(-1,1)),(1,T)))                # Posterior expectation of all hidden states at the current time
        layer_input.X_archive.append(np.tile(np.reshape(d[f],(-1,1,1)),(T,T)))      # Estimation at time t of BMA states at time tau
        layer_input.S.append(np.zeros((Ns[f],T,T)) + 1./Ns[f])                      # Posterior expectation of all hidden states over time
        for k in range(Np):
            layer_input.x[f][:,0,k] = d[f]

    layer_input.Q = []
    for t in range(T):
        layer_input.Q.append(spm_kron(D))


    layer_input.vn = []
    for f in range(Nf):        
        layer_input.vn.append(np.zeros((Ni,Ns[f],T,T,Np)))  # Recorded neuronal prediction error
    
    #ACTIONS ------------------------------------------------------------------
    # >>> Action_selection

    #history of posterior over action
    layer_input.u_posterior_n = np.zeros((Np,Ni*T))             
    #posterior over action
    layer_input.u_posterior = np.zeros((Np,T))                
    
    # >>> Chosen Action
    u_temp = np.full((Nf,T-1),-1)  
    if isField(layer_input.u_):
        u_temp[layer_input.u_>=0] = layer_input.u_[layer_input.u_>=0]
    layer_input.u = u_temp


    layer_input.K = np.full((T-1,),-1)
    #POLICIES ------------------------------------------------------------------
    #history of posterior over policy
    layer_input.p_posterior_n = np.zeros((Np,Ni*T))             
    #posterior over policy
    layer_input.p_posterior = np.zeros((Np,T))                 
    if (Np == 1) :
        layer_input.p_posterior = np.ones((Np,T))
            
    # Allowable policies
    p = np.zeros((Np,))
    for policy in range(Np): # Indices of allowable policies
        p[policy] = policy
    layer_input.p = p.astype(np.int)


    #Posterior over policies & actions
    layer_input.P = np.zeros((tuple(Nu)+(1,)))


    # -------------------------------------------------------------
    #TODO : Initialize output variables for a run
    layer_input.L = []


    layer_input.F = np.zeros((layer_input.Np,layer_input.T))
    layer_input.G = np.zeros((layer_input.Np,layer_input.T))
    layer_input.H = np.zeros((layer_input.T,))     
    



    layer_input.w = np.zeros((T,))  # Policy precision w
    layer_input.wn = None          # Neuronal encoding of policy precision

    layer_input.dn = None          # Simulated dopamine response
    layer_input.rt = np.zeros((T,))          # Simulated reaction times
    
    

    init_precisions(layer_input)



def belief_update(cache):
    # t : timestep in which the update happens
    # o : array of observations up to t
    # u : array of actions up to t-1
    # mdp parameters : [a,b,c,d,e]
    # q_s : history of state estimates
    # q_u : history of action estimates
    Np = self.Np
    Nu = self.Nu

    Ni = self.options.Ni
    t = self.t
    T = self.T

    N = self.options.T_horizon

    t0 = time.time()

    # Prior over subsequent states
    if (t>0):
        q_s[t] = np.dot(self.b_kron[u[t-1]],q_s[t-1])
    else :
        q_s[t] = spm_kron(self.d)

    reduced_O = []
    for mod in range(Nmod):
        reduced_O.append(self.O[mod][:,t])

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





    return q_s,q_pi