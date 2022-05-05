import numpy as np
from base.function_toolbox import normalize,softmax
from base.miscellaneous_toolbox import isField

class tree_node :
    def __init__(self,deterministic_ = True,time=0):
        self.t = time
        self.data = None
        self.parent = None
        self.deterministic = deterministic_

        self.states = None  # Array, flattened prob distribution of expected states. 
        self.action_density = None # Float,  Q(choosing this state | previously chosen state)
        self.u = None # Float, Q(being in this state at time t)
        self.children_nodes = []

    def has_children(self):
        return (len(self.children_nodes)>0)

    def add_child(self, child_state_dist, child_action_weight) :
        child_node = tree_node((child_action_weight==1))
        child_node.states = np.copy(child_state_dist)
        child_node.action_density = child_action_weight
        child_node.t = self.t + 1
        child_node.parent = self
        self.children_nodes.append(child_node)
    
    def add_child_node(self, child_node) :
        child_node.parent = self
        child_node.t = self.t + 1
        self.children_nodes.append(child_node)
    
    def add_children(self,list_of_child_states_depending_on_action,array_of_action_weights):
        for k in range(len(list_of_child_states_depending_on_action)): # Considered actions
            self.add_child(list_of_child_states_depending_on_action[k],array_of_action_weights[k])

    def compute_subsequent_state_prob(self):
        """ Assumes we know u for this node, compute u for subsequent nodes"""
        if (self.parent == None):
            self.u = 1.0
        for node in (self.children_nodes):
            node.u = self.u * node.action_density
            node.compute_subsequent_state_prob()
            
    def get_descendants(self,including_self=False):
        R = []
        if (including_self):
            R.append(self)
        
        if (len(self.children_nodes)==0):
            return R
        else :
            for k in range(len(self.children_nodes)):
                R = R + self.children_nodes[k].get_descendants(True)
            return R
    
    def to_string(self,passage_at_t=-1):
        tabs = "     "
        last_tabs = "|----"
        passage_tabs = "|    "
        message = str(self.t) + " "
        for i in range(self.t-1):
            for k in passage_at_t:
                if (i==k):
                    message += passage_tabs
                    break
            else :
                message = message + tabs
        if (self.t > 0):
            message = message + last_tabs
        if not(isField(self.data)):
            self.data = "X"
        message = message + str(self.data)
        message = message + "\n"
        return message

    def tostringfamily(self,passage_at_t=None,generations_remaining = None):
        if(passage_at_t==None):
            passage_at_t = []
        me = self.to_string(passage_at_t)

        if not(self.has_children()) or (generations_remaining==0):
            return me
        else :
            if (generations_remaining != None):
                generations_remaining = generations_remaining - 1
            for child in range(len(self.children_nodes)) :
                if not(child == len(self.children_nodes)-1):
                    me = me + self.children_nodes[child].tostringfamily(passage_at_t + [self.t],generations_remaining)
                else :
                    me = me + self.children_nodes[child].tostringfamily(passage_at_t,generations_remaining)
            return me

class state_tree :
    def __init__(self,initial_node):
        self.root_node = initial_node
    
    def get_all_nodes(self):
        return (self.root_node.get_descendants(True))

    def get_nodes(self, at_time):
        """ return a list of nodes at time at_time"""
        L = self.root_node.get_descendants(True)
        R = []
        for node in L:
            if (node.t == at_time):
                R.append(node)
        return R
    
    def get_expected_states(self,at_time):
        L = self.get_nodes(at_time)
        if (len(L)==0):
            # We are beyond the existing temporal horizon --> return a flattened expectation ?
            return normalize(np.ones(self.root_node.states.shape))
        R = (np.zeros((L[0].states.shape)))
        for node in L:
            R = R + node.u * node.states
        return R
    
    def matrix_of_state_expectations(self,t0,t1):
        """ t1 not included, Python style :D"""
        self.root_node.compute_subsequent_state_prob()
        R = np.zeros((self.root_node.states.shape) + (t1 - t0,))

        corrector_initial = 0
        if (t0 == -1):
            corrector_initial = 1   # If we want to provide state expectations before the beginning of the experience, we have to introduce a shift to the matrix indices
        for ti in range (t0,t1):
            R[:,ti + corrector_initial] = self.get_expected_states(ti)
        return R

    def to_string(self,generations_remaining_=None):
        return self.root_node.tostringfamily(generations_remaining=generations_remaining_)
