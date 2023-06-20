from ..base.miscellaneous_toolbox import isField,flexible_copy
BASIC_SPACE = "   "

class policy_tree_node :
    """ For now, just used to give an account of the computations made by the agents when selecting 
    actions. At some point, should be used to implement spm_forwards in a more object-oriented fashion.
    (each node called recursively, better control over the calculations, etc.)"""

    def __init__(self,deep_index,s_prior,n_possible_policies,n_possible_states,
                 u_index = 0,s_index=0):
        self.deep_index = deep_index

        self.u_index = u_index
        self.s_index = s_index

        self.state_prior = flexible_copy(s_prior)
        self.state_posterior = None
        self.policy_prior = None
        self.policy_posterior = None
        self.policy_weighted_next_state_posterior = None
        self.n_children = [n_possible_policies,n_possible_states]
        self.children = []
        for k in range(n_possible_policies):
            self.children.append([])
            for j in range(n_possible_states):
                self.children[k].append(None)
        # print(n_possible_policies,n_possible_states)
    
    def update_policy_posterior(self,pp):
        self.policy_posterior = flexible_copy(pp)
    
    def update_policy_prior(self,pp):
        self.policy_prior = flexible_copy(pp)
    
    def update_state_posterior(self,sp):
        self.state_posterior = flexible_copy(sp)

    def update_pol_weighted_next_state_posterior(self,sp):
        self.policy_weighted_next_state_posterior = flexible_copy(sp)
    
    def add_child(self,child_state_prior,u_index_in = 0,s_index_in =0):
        self.children[u_index_in][s_index_in] = policy_tree_node(self.deep_index+1,child_state_prior,self.n_children[0],self.n_children[1],u_index_in,s_index_in)
        return self.children[u_index_in][s_index_in]
    
    def get_tree_str(self,preprend=""):
        def get_line(content):
            if(not(isField(content))):
                content = "N.A"
            return preprend + str(content) + "\n"
        

        tree_str=""
        # 1. print my tree : 
        # tree_str += get_line("Index :  - "+ str(self.deep_index))
        
        # ME : 
        tree_str += get_line("Level : "+str(self.deep_index)+"  - Action : " +str(self.u_index) + " leading to state " + str(self.s_index))
        # tree_str += get_line(self.policy_weighted_next_state_posterior)

        for child_act in self.children:
            for child in child_act:
                # child_action_values = self.value[child_idx]
                # tree_str += get_line("- Child : "+str(child_idx) + " (lev" + str(self.deep_index+1) + ") -- action " + str(self.u_index) + " + leading to state " + str(self.s_index))
                if (isField(child)):
                    # child_prob_values = self.value[child_idx]
                    tree_str += child.get_tree_str(preprend+BASIC_SPACE)
        return tree_str

    def hasChildren(self):
        for action_children in self.children:
            for child in action_children:
                if (isField(child)):
                    return True
        return False
    
    def explored_paths(self):
        # One branche is a list of size (T_horizon + 1)
        # The result of this function is a list of lists 
        to_return = (str(self.u_index)+"_"+str(self.s_index))
        if (not(self.hasChildren())):
            return [[to_return]]
        else : 
            paths = []
            for action_children in self.children:
                for child in action_children:
                    if (isField(child)):
                        for path in child.explored_paths():
                            paths.append([to_return] + path)
            return paths
        
   

   