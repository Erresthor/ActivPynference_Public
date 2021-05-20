# -*- coding: utf-8 -*-
"""
Created on Wed May 19 16:00:55 2021

@author: cjsan
"""
import numpy as np
import matplotlib.pyplot as plt
from spm12_implementation import MDP
from explore_exploit_model import explore_exploit_model
from plotting_toolbox import basic_autoplot

class VB_agent:
    """Let's define an agent as a family of MDPs. (different generations of the same MDP)"""
    
    def __init__(self):
        self.MDP_list = []
        self.NumGens = 0
        
        self.trained = False
    
    def train(self,mdp,NumGens):
        self.NumGens = NumGens
        if (not(self.trained)):
            self.MDP_list = mdp.run_for_N_generations(NumGens)
            self.trained= True
    
    def show(self):
        assert (self.trained),"Nothing to show : this agent hasn't trained"
        Ng = self.NumGens
        
        
        X = np.linspace(1,Ng,Ng)
        Chosen_first_action = np.zeros((Ng,))
        Chosen_second_action = np.zeros((Ng,))
        Outcome = np.zeros((Ng,))
        Context_state_at_t0 = np.zeros((Ng,)) #
        True_context_state = np.zeros((Ng,))
        
        
        Ni = 16
        T=3
        Precision = np.zeros((Ng*Ni*T,))
        Dopamine = np.zeros((Ng*Ni*T,))

        
        for i in range(Ng):
            
            Chosen_first_action[i] = self.MDP_list[i].u[1,0]
            Chosen_second_action[i] = self.MDP_list[i].u[1,1]
            
            #Context state (which side is the best ?)
            Context_state_at_t0[i] = 1-self.MDP_list[i].xn[0][-1,0,0,0,0] + 2
            True_context_state[i] = self.MDP_list[i].s[0,0] + 2
            
            print(self.MDP_list[i].wn.shape)
#            print(self.MDP_list[i].X[0].shape)
#            print(self.MDP_list[i].Q[0])
            Outcome[i] = np.max(self.MDP_list[i].o[1,:])
            
            Precision[i*Ni*T:(i+1)*Ni*T] = self.MDP_list[i].wn*(self.MDP_list[i].wn>0)
            Dopamine[i*Ni*T:(i+1)*Ni*T] = self.MDP_list[i].dn*(self.MDP_list[i].dn>0)
        fig,ax = plt.subplots()
        
        Action_ticks = ['Inactive','Hint','Left','Right']
        ax.plot(X,Context_state_at_t0,label='belief about context state',color='r')
        ax.plot(X,True_context_state,label='true context state',color='g')
        ax.scatter(X,Chosen_first_action,label='first action')
        ax.scatter(X,Chosen_second_action,label='second action')
        ax.set_yticks([0,1,2,3])
        ax.set_yticklabels(Action_ticks)
        plt.legend()
        plt.show()
        
        plt.figure()
        basic_autoplot(Precision)
        
        plt.figure()
        basic_autoplot(Dopamine)
        #plt.scatter(X,Outcome,label='Outcome')
        #plt.yticks([0,1,2],['Undecided', 'Win' ,'Loss'],size="small")
        #plt.show()
        
        print(str(100*(1-np.sum(Outcome-1)/Ng)) + " % of wins")

agent = VB_agent()
agent.train(explore_exploit_model(0.8,pWin=1,pHA=1,rs=3,la=1),30)
agent.show()