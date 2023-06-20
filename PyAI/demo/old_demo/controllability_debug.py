import sys,os

import numpy as np
import matplotlib.pyplot as plt

from pyai.base.function_toolbox import *
from pyai.base.miscellaneous_toolbox import flexible_copy

from pyai.model import active_model,active_model_container,active_model_save_manager
from pyai.neurofeedback_run import save_model_performance_dictionnary,load_model_performance_dictionnary
from pyai.layer.layer_learn import MemoryDecayType
from pyai.layer.mdp_layer import mdp_layer

from pyai.base.function_toolbox import normalize,spm_dot, nat_log,softmax
from pyai.base.miscellaneous_toolbox import isField
from pyai.base.miscellaneous_toolbox import isNone,flatten_last_n_dimensions,flexible_toString,flexible_print,flexible_copy

from pyai.model.metrics import matrix_kl_dir

def controlability_task_layer(mem_dec_term=100,rule = 'c1',lr=1):
    Nf = 1

    initial_state = 0
    D_ = normalize([np.array([1,1,1])])         #[Square, Triangle, Circle]
    d_ = [0.1*np.ones(D_[0].shape)]


    A_ = [np.eye(3)] # The observations are the current shape
    # No a_, the observations are perfect

    pb = 0.05
    
    # Line = where we're going
    # Column = where we're from
    B_ = [np.zeros((3,3,3))] # 3 from-states, 3 to-states, 3 possible actions (yellow, blue, purple)

    state_dependent_allowable_action = np.array([[1      ,0         ,1],     # Yellow
                                                [0      ,1         ,1],     # Blue
                                                [1      ,1         ,0]])   # Purple
                                                # Square | Triangle | Circle

    # state_dependent_allowable_action = None

    if (rule=='c1'):
        # CONTROLABLE RULE 1
        # Action 1 : yellow -> triangle
        B_[0][:,:,0] = np.array(
                [[pb    ,pb    ,pb    ],
                 [1-2*pb,1-2*pb,1-2*pb],
                 [pb    ,pb    ,pb    ]]
        )
        # Action 2 : blue -> square
        B_[0][:,:,1] = np.array(
               [[1-2*pb,1-2*pb,1-2*pb],
                [pb    ,pb    ,pb    ],
                [pb    ,pb    ,pb    ]]
        )
        # Action 3 : purple -> circle
        B_[0][:,:,2] = np.array(
               [[pb    ,pb    ,pb    ],
                [pb    ,pb    ,pb    ],
                [1-2*pb,1-2*pb,1-2*pb]]
        )
    elif (rule=='c2'):
        # CONTROLABLE RULE 2
        # Action 1 : yellow -> square
        B_[0][:,:,0] = np.array(
                [[1-2*pb,1-2*pb,1-2*pb],
                    [pb    ,pb    ,pb    ],
                    [pb    ,pb    ,pb    ]]
        )
        # Action 2 : blue -> circle
        B_[0][:,:,1] = np.array(
                [[pb    ,pb    ,pb    ],
                    [pb    ,pb    ,pb    ],
                    [1-2*pb,1-2*pb,1-2*pb]]
        )
        # Action 3 : purple -> triangle
        B_[0][:,:,2] = np.array(
                [[pb    ,pb    ,pb    ],
                    [1-2*pb,1-2*pb,1-2*pb],
                    [pb    ,pb    ,pb    ]]
        )

        
    elif (rule=='u1'):
        # UNCONTROLABLE RULE 1
        # Action 1 : yellow -> square
        B_[0][:,:,0] = np.array(
                [[pb    ,pb    ,1-2*pb],
                 [1-2*pb,pb    ,pb    ],
                 [pb    ,1-2*pb,pb    ]]
        )
        B_[0][:,:,1] = np.array(
                [[pb    ,pb    ,1-2*pb],
                 [1-2*pb,pb    ,pb    ],
                 [pb    ,1-2*pb,pb    ]]
        )
        B_[0][:,:,2] = np.array(
                [[pb    ,pb    ,1-2*pb],
                 [1-2*pb,pb    ,pb    ],
                 [pb    ,1-2*pb,pb    ]]
        )

    elif (rule=='u2'):
        # UNCONTROLABLE RULE 2
        # Action 1 : yellow -> square
        B_[0][:,:,0] = np.array(
                [[pb    ,1-2*pb,pb    ],
                 [pb    ,pb    ,1-2*pb],
                 [1-2*pb,pb    ,pb    ]]
        )
        B_[0][:,:,1] = np.array(
                [[pb    ,1-2*pb,pb    ],
                 [pb    ,pb    ,1-2*pb],
                 [1-2*pb,pb    ,pb    ]]
        )
        B_[0][:,:,2] = np.array(
                [[pb    ,1-2*pb,pb    ],
                 [pb    ,pb    ,1-2*pb],
                 [1-2*pb,pb    ,pb    ]]
        )
    else : 
        print("Indicated rule " + rule + " hasn't been implemented")
        return
    B_ = normalize(B_)
    b_ = [np.ones((B_[0].shape))]
    

    C_ = [np.array([[1],
                   [1],
                   [1]])]


    # Policies
    U_ = np.array([
        [0],
        [1],
        [2]
    ])

    #Habits
    E_ = None

    T = 7



    layer = mdp_layer()
    
    layer.T = T
    layer.A_ = A_
    layer.B_ = B_
    layer.C_ = C_
    layer.D_ = D_

    layer.b_ = b_
    layer.d_ = d_
    
    layer.U_ = U_
    layer.state_u = state_dependent_allowable_action

    layer.options.learn_a = False
    layer.options.learn_b = True
    layer.options.learn_d = False
    layer.options.T_horizon = 2
    layer.options.learn_during_experience = False
    layer.options.memory_decay = MemoryDecayType.NO_MEMORY_DECAY
    layer.options.memory_decay = MemoryDecayType.STATIC
    layer.options.decay_half_time = mem_dec_term

    layer.parameters.eta = lr
    return layer

def state(idx):
    if (idx==0):
        return "Square"
    if (idx==1):
        return "Triangle"
    if (idx==2):
        return "Circle"

def action(idx):
    if (idx==0):
        return "Yellow"
    if (idx==1):
        return "Blue"
    if (idx==2):
        return "Purple"

def duplicate_a_layer(layer_example,copy_num):
    list_return = []
    for cop in range(copy_num):
        list_return.append(layer_example.copy())
    return list_return

def run_n_layer_trials(layer,n,first_trial_is_new_rule=False):
    model_errors = []
    for trial in range(n):
        if (first_trial_is_new_rule and (trial==0)):
            model_errors.append(np.sum(np.abs(normalize(layer.b_[0])-layer.B_[0])))

        layer.prep_trial()
        for t in range(layer.T):
            layer.tick()
        layer.postrun(True)

        # Calculate the model transition error term : 
        # diff = np.abs(normalize(layer.b_[0])-layer.B_[0])
        # allowable_only = layer.state_u
        # total_diff = 0
        # for state_from in range(diff.shape[0]):
        #     for state_to in range(diff.shape[1]):
        #         for action in range(diff.shape[2]):
        #             if allowable_only[action,state_from]:
        #                 total_diff += diff[state_from,state_to,action]

        # print(total_diff)
        model_errors.append(np.sum(np.abs(normalize(layer.b_[0])-layer.B_[0])))
        #model_errors.append(matrix_kl_dir(normalize(layer.b_[0]),layer.B_[0]))

    return layer, model_errors

def return_result_array(mem_dec_term,same_rule_may_follow = False,
    Ntrials = 20 , Ninstances = 10, Nsections = 10,
    override_rules=None,lr=1,override_prompts=None):  
    
    # Initialize controlability layers
    if isField(override_rules):
        rule = override_rules[0]
    else : 
        rule = random.choice(['c1','c2','u1','u2'])
    model_layer = controlability_task_layer(mem_dec_term=mem_dec_term,rule=rule,lr=lr)
    lay_list = duplicate_a_layer(model_layer,Ninstances)
    
    # Initialize lists
    errors = []
    # for lay in lay_list :
    #     errors.append([np.sum(np.abs(normalize(lay.b_[0])-lay.B_[0]))])
    rule_history = [rule]

    errors =[]
    for lay in lay_list :
        errors.append([])
    rule_history=[]
    # Run the experiment
    predictions = []
    for i in range(Nsections):
        print("Section " + str(i))
        print("Associated rule : " + str(rule))
        rule_history_section = []
        for k in range(Ntrials+1):
            rule_history_section.append(rule)
        rule_history += rule_history_section

        # Update B values
        intermediate_controlability_layer = controlability_task_layer(mem_dec_term=mem_dec_term,rule=rule,lr=lr)
        for nlay in range(Ninstances):
            lay_list[nlay].B_ = flexible_copy(intermediate_controlability_layer.B_)
        
        # for k in range(Ninstances):
        #     print(rule)
        #     print(intermediate_controlability_layer.B_)
        #     print(lay_list[k].B_)
        # Run inferences
        for nlay in range(Ninstances):
            lay, model_error = run_n_layer_trials(lay_list[nlay],Ntrials,first_trial_is_new_rule=True)
            errors[nlay] += model_error

        # Pick new rule for the next iteration
        if isField(override_rules)and(i<Nsections-1):
            rule = override_rules[i+1]
        else : 
            if (same_rule_may_follow):
                rule = random.choice(['c1','c2','u1','u2'])
            else : 
                rule_candidate = random.choice(['c1','c2','u1','u2'])
                while(rule_candidate==rule):
                    rule_candidate = random.choice(['c1','c2','u1','u2'])
                rule = rule_candidate

        if (isField(override_prompts)):
            for nlay in range(Ninstances):
                print("For rule " + str(rule))
                Q,Q0 = predict_upcoming_obs_given_action(override_prompts[i][0][0],normalize(np.ones((3,))),lay_list[nlay])
                for act in range(3):
                    if isField(lay_list[nlay].state_u):
                        if (lay_list[nlay].state_u[act,override_prompts[i][0][0]]):
                            print("For action "+ str(act) + " :")
                            print(Q0[act])
                    else :
                        print("For action "+ str(act) + " :")
                        print(Q0[act])
                print("-----------")
    return errors,rule_history

def predict_upcoming_obs_given_action(last_O,Prior_P,layer,verbose = False) :    
    # Nf = len(B) --> not because B is in Kronecker form
    layer.prep_trial()
    a = layer.a
    b = layer.b_kron
    true_B = layer.B
    U = layer.U_
    # get kronecker form of B : 
    true_B_kron = []
    for action in range(U.shape[0]) :
        true_B_kron.append(1)
        for factor in range(len(true_B)):
            true_B_kron[action] = spm_kron(true_B_kron[action],true_B[factor][:,:,U[action,factor]])

    true_A = layer.A
    U = layer.U
    
    Nmod = len(a)
    Prior_P = np.copy(Prior_P)

    # SUBJECT STATE / NEXT STATE GIVEN ACTION INFERENCE
    # L is the posterior over hidden states based on last observations & likelihood (A & O)
    L = 1
    for modality in range (Nmod):
        L = L * spm_dot(a[modality],last_O) 
    P_posterior =normalize(L.flatten()*Prior_P) # P(s|o) = P(o|s)P(s)  
  
    Q = []
    QO = []
    for action in range(U.shape[0]) :
        Q.append(np.dot(b[action],P_posterior)) # predictive posterior of states at time t depending on actions 
        QO.append(1)
        for modality in range(Nmod):
            flattened_A = flatten_last_n_dimensions(a[modality].ndim-1,a[modality])
            qo = np.dot(flattened_A,Q[action]) # prediction over observations at time t
            QO[action] = qo
    
    # PROCESS STATE / NEXT STATE GIVEN ACTION INFERENCE
    L = 1
    for modality in range (Nmod):
        L = L * spm_dot(true_A[modality],last_O) 
    process_posterior = normalize(L.flatten()*Prior_P) # P(s|o) = P(o|s)P(s)  
    
    proc_Q = []
    proc_QO = []
    for action in range(U.shape[0]) :
        proc_Q.append(np.dot(true_B_kron[action],process_posterior)) # predictive posterior of states at time t depending on actions 
        proc_QO.append(1)
        for modality in range(Nmod):
            flattened_A = flatten_last_n_dimensions(true_A[modality].ndim-1,true_A[modality])
            procqo = np.dot(flattened_A,proc_Q[action]) # prediction over observations at time t
            proc_QO[action] = procqo
    
    verbose = False
    if verbose :
        print("POSTERIORS OVER STATES AT TIME t")
        print("Process :")
        print(process_posterior)
        print("Model :")
        print(P_posterior)
        print()
        print("ACTION DEPENDENT POSTERIOR AT TIME t+1")
        print("Process :")
        print(np.round(proc_QO,2))
        print("Model :")
        print(np.round(QO,2))

    # Q = Next state given action dist, QO = next obs given action dist, 
    return [Q,QO],[proc_Q,proc_QO]

def controlability_model_perf_plot(override_rules=None,override_prompts=None):
    mem_dec_list = [0.1,0.5,1,1.5,2.5,5,1000]
    learn_rate_list = [1]
    Ntrials = 10
    Ninstances = 10
    if (isField(override_rules)):
        Nsections = len(override_rules)
    else :
        Nsections = 10
    
    if len(learn_rate_list)>1:
        fig,axes = plt.subplots(len(mem_dec_list),len(learn_rate_list))
    else :
        fig,axes = plt.subplots(len(mem_dec_list),1)
    cnt = [0,0]

    
    for learn_rate in learn_rate_list:
        for mem_dec_term in mem_dec_list:
            
            errors,rule_history = return_result_array(mem_dec_term,same_rule_may_follow=True,Ntrials= Ntrials,Ninstances=Ninstances,Nsections=Nsections,
                                override_rules=override_rules,lr=learn_rate,
                                override_prompts=override_prompts)
            if (len(learn_rate_list)>1):
                ax = axes[tuple(cnt)]
            else : 
                ax = axes[cnt[0]]
            ax.grid()
            ax.axvline(0,color='grey')
            ax.text(Ntrials/4, 12, rule_history[0], fontsize=20)
            for t in range(1,len(rule_history)) :
                if rule_history[t] != rule_history[t-1] :
                    ax.axvline(t,color='grey')
                    ax.text(t+Ntrials/4, 12, rule_history[t], fontsize=20)
            for i in range(Ninstances):
                ax.scatter(np.arange(0,len(errors[i]),1),errors[i],color=np.array([1.0,0.6,0.6,0.25]),s=0.5)
            # ax.plot(np.arange(0,len(errors[i]),1),np.mean(np.array(errors),axis=0),color='red',linewidth=2)
                        
            ax.scatter(np.arange(0,len(errors[i]),1),np.mean(np.array(errors),axis=0),color='red',s=5)

            ax.set_ylim([0.0,15])

            cnt[0] = cnt[0]+1
        cnt[1] = cnt[1] + 1
        cnt[0] = 0
    fig.show()
    plt.show()

def change_layer_Bprocess(layer,newrule):
    model_layer_int = controlability_task_layer(mem_dec_term=5,rule=newrule)
    layer.B_ = flexible_copy(model_layer_int.B_)

def run_and_predict(layer,last_o,run_n=1):
    layer,model_errs = run_n_layer_trials(layer,run_n)
    prior_P = normalize(np.array([1,1,1]))
    QQO,Proc_QQO = predict_upcoming_obs_given_action(last_o,prior_P,layer)
    QO = QQO[1] #print(QO)
    proc_QO = Proc_QQO[1]
    return QO,QQO[0],proc_QO,model_errs

def run_predict_and_update(layer,pred_s_idx,run_n=1,
                lr_win = 1,lr_lose = -1,
                first_feedback_known=True,second_feedback_known=False):
    
    prediction_state = np.zeros((3,))
    prediction_state[pred_s_idx] = 1
    
    QO,Q,proc_QO,model_errs = run_and_predict(layer,prediction_state,run_n)
    
    # First, only allowable actions are prompted
    allowable_act = []
    for action in range(3):
        if layer.state_u[action,pred_s_idx] :
            allowable_act.append(action)
    random.shuffle(allowable_act) # Randomize the first color asked so that no biaises 

    # Actual prompt :
    prompt_k = 0
    feedbacks = []
    for action_prompted in allowable_act :

        gamma_pred = 1
        # subject_next_obs_predict = cat(softmax(gamma_pred*QO[action_prompted]))
        subject_next_obs_predict = cat(QO[action_prompted])

        gamma = 100 # gamma in [1,+oo], depending on how deterministic the feedback is 
                    # [ 1 : true transition is picked randomly, +oo, feedback is picked depending on the maximum probability transition]
        proc_next_obs_predict = cat(softmax(gamma*proc_QO[action_prompted]))

        feedback = (subject_next_obs_predict==proc_next_obs_predict)
        feedback_int = (1 if feedback else 0)
        reinforcment_term = (lr_win if feedback else lr_lose)

        # Is feedback known to the subject ?
        if first_feedback_known and prompt_k==0:
            # What impact of the subject model ?
            # If the feedback is good, reinforce the couple 
            #      ( action, state from , state to) by adding a *mlr multiplicative constant (>1)
            # Else diminish it : 
            layer.b_[0][pred_s_idx,subject_next_obs_predict,action_prompted] = max(layer.b_[0][pred_s_idx,subject_next_obs_predict,action_prompted] + reinforcment_term,1e-3)
                                                            # Prevent negative values
            # layer.b_[0] = normalize(layer.b_[0])
        if second_feedback_known and prompt_k==1:
            layer.b_[0][pred_s_idx,subject_next_obs_predict,action_prompted] = max(layer.b_[0][pred_s_idx,subject_next_obs_predict,action_prompted] + reinforcment_term,1e-3)
            # layer.b_[0] = normalize(layer.b_[0])
        
        feedbacks.append([subject_next_obs_predict,proc_next_obs_predict,feedback_int])
        prompt_k += 1
    
    return model_errs,feedbacks


def add_random_prompts(tolist,n=2):
    L = []
    for k in range(n):
        prediction_state = np.zeros((3,))
        prediction_state[random.choice([0,1,2])] = 1
        prediction_action = random.choice([0,1,2])
        L.append([prediction_state,prediction_action])
    tolist.append(L)

def cat(one_d_dist):
    N = one_d_dist.shape[0]
    return np.argwhere(random.random() <= np.cumsum(one_d_dist))[0][0]
    #while random_01 < np.cumsum(one_d_dist)[k]:

def run_controlability_trials(mdt=1,all_same = False,no_consec = True):
    xs = []
    errs_lis = []
    vertical_lines_xs = []
    rule_history = []
    feedback_history = []

    # Generate forthcoming transition rules
    possible_choices = ['c1','c2','u1','u2']
    rules = []
    for subject in range(Nsubjects):
        rules.append([])
        if (subject > 0) and all_same:
            for sect in range(Nsect):
                rules[subject].append(rules[subject-1][sect])
        else :
            rules[subject].append(random.choice(possible_choices))
            for sect in range(Nsect-1):
                potent_choix = random.choice(['c1','c2','u1','u2'])
                if no_consec:
                    while(potent_choix == rules[subject][sect]) :
                        potent_choix = random.choice(['c1','c2','u1','u2'])
                rules[subject].append(potent_choix)
    
    n_sections = len(rules[0])
    n_subjs = len(rules)

    model_layer_ex = controlability_task_layer(mem_dec_term=1,rule='c1')
    lay_list = duplicate_a_layer(model_layer_ex,n_subjs)
    for lay_idx in range(len(lay_list)):
        print("Subject " + str(lay_idx+1) + " / " + str(len(lay_list)))
        subj_idx = lay_idx
        model_layer = lay_list[lay_idx]
        
        if (type(mdt)==list or (type(mdt)==np.ndarray)):
            model_layer.options.decay_half_time = mdt[lay_idx]
        else : 
            model_layer.options.decay_half_time = mdt
        
        errs_lis.append([])
        vertical_lines_xs.append([])
        rule_history.append([])
        feedback_history.append([])
        vert_cnt=0

        for section in range(n_sections):
            rule = rules[subj_idx][section]
            # print("CHANGE TRANSITION RULE TOWARDS : " + rule)
            
            change_layer_Bprocess(model_layer,rule)
            errs_lis[lay_idx].append(np.sum(np.abs(normalize(model_layer.b_[0])-model_layer.B_[0])))
            
            # Count the moments the rule changed 
            vertical_lines_xs[lay_idx].append(vert_cnt)
            vert_cnt += 1


            n_trials_with_this_rule = 0
            last_feedbacks = [[0,0],[0,0,0,0]] # [in the last 4 trials / in the last 6 trials]
            continue_with_this_rule = True 
            while continue_with_this_rule:
                n_trials_with_this_rule += 1
                rule_history[lay_idx].append(rule)          
                
                # Which next state is expected by the subject ?
                # First, from which state do we start for the prediction ?
                state_i = random.choice([0,1,2])

                # Actual run : "training" + predictions for a given starting state
                errs,feedbacks = run_predict_and_update(model_layer,state_i,run_n = run_n)
                
                
                pred_1 = feedbacks[0][2] # 1 or 0 depending on how good was the feedback
                pred_2 = feedbacks[1][2] # 1 or 0 depending on how good was the feedback

                last_feedbacks[0].append(pred_1)
                last_feedbacks[1].append(pred_1)
                last_feedbacks[0].append(pred_2)
                last_feedbacks[1].append(pred_2)                    
                
                if n_trials_with_this_rule>=4 :
                    if (sum(last_feedbacks[0])==4) or (sum(last_feedbacks[1])>=5) :
                        continue_with_this_rule = False
                errs_lis[lay_idx] += errs
                feedback_history[lay_idx].append([pred_1,pred_2])
                vert_cnt += run_n
                
                last_feedbacks[0].pop(0)
                last_feedbacks[0].pop(0)
                last_feedbacks[1].pop(0)
                last_feedbacks[1].pop(0)
    
    return errs_lis,vertical_lines_xs,rule_history,feedback_history

def predictive_accuracy_around_rulechange(rule_history,feedback_history):
    # Record the predictive performances of the agents when a rule change occurs
    prediction_quality_when_rule_changes = [[],[],[],[]]
    # 1: cc , 2:cu, 3:uu , 4:uc

    for subj_idx in range(len(rule_history)):
        for trial_k in range(1,len(rule_history[subj_idx])):
            previous_rule = rule_history[subj_idx][trial_k-1]
            next_rule = rule_history[subj_idx][trial_k]
            
            if (previous_rule != next_rule) :
                L = feedback_history[subj_idx][trial_k-2:trial_k+4]
                # print(L)
                if ("c" in previous_rule) and ("c" in next_rule):
                    prediction_quality_when_rule_changes[0].append(L)
                elif ("c" in previous_rule) and ("u" in next_rule):
                    prediction_quality_when_rule_changes[1].append(L)
                elif ("u" in previous_rule) and ("u" in next_rule):
                    prediction_quality_when_rule_changes[2].append(L)
                elif ("u" in previous_rule) and ("c" in next_rule):
                    prediction_quality_when_rule_changes[3].append(L)
                else : 
                    print(previous_rule,next_rule)
                    print("What the fk ?")

    # print(np.array(prediction_quality_when_rule_changes[0]))
    array_cc = np.array(prediction_quality_when_rule_changes[0])
    array_cu = np.array(prediction_quality_when_rule_changes[1])
    array_uu = np.array(prediction_quality_when_rule_changes[2])
    array_uc = np.array(prediction_quality_when_rule_changes[3])

    return array_cc,array_cu,array_uu,array_uc

def fig_around_rulechange_acc(array_cc,array_cu,array_uu,array_uc) :
    xs = np.array([-2,-1,1,2,3,4])
    k = 0.5
    fills_transparency = 0.1

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.grid()
    ax.axhline(0.33333,color='grey')
    ax.axvline(0,color='grey')
    mean_uu = np.mean(array_uu,axis=(0,2))
    std_uu = np.std(array_uu,axis=(0,2))
    ax.fill_between(xs,mean_uu-k*std_uu,mean_uu+k*std_uu,color=np.array([1,0,0,fills_transparency]))
    ax.plot(xs,mean_uu,color=np.array([1,0,0,1]))
    
    mean_cu = np.mean(array_cu,axis=(0,2))
    std_cu = np.std(array_cu,axis=(0,2))
    ax.fill_between(xs,mean_cu-k*std_cu,mean_cu+k*std_cu,color=np.array([0,0,1,fills_transparency]))
    ax.plot(xs,mean_cu,color=np.array([0,0,1,1]))

    mean_uc = np.mean(array_uc,axis=(0,2))
    std_uc = np.std(array_uc,axis=(0,2))
    ax.fill_between(xs,mean_uc-k*std_uc,mean_uc+k*std_uc,color=np.array([0,1,0,fills_transparency]))
    ax.plot(xs,mean_uc,color=np.array([0,1,0,1]))
    
    mean_cc = np.mean(array_cc,axis=(0,2))
    std_cc = np.std(array_cc,axis=(0,2))
    ax.fill_between(xs,mean_cc-k*std_cc,mean_cc+k*std_cc,color=np.array([0,0.8,0.8,fills_transparency]))
    ax.plot(xs,mean_cc,color=np.array([1,0.65,0,1]))
    # plt.plot(xs,np.mean(array_cc,axis=(0,2)),color='yellow')
    ax.set_ylim([0,1])
    return fig

if __name__=="__main__":
    verbose = False

    Nsect = 20
    Nsubjects = 5
    run_n = 1 # Number of trials before predictions, normally 1
    # mdt = np.linspace(0.3,20,Nsubjects)
    mdt = 8
    for mdt in [0.25]:
        errs_lis,vertical_lines_xs,rule_history,feedback_history = run_controlability_trials(mdt,all_same=False,no_consec=True)
        
        array_cc,array_cu,array_uu,array_uc = predictive_accuracy_around_rulechange(rule_history,feedback_history)

        fig = fig_around_rulechange_acc(array_cc,array_cu,array_uu,array_uc)
        fig.show()
    plt.show()


        # n = len(errs_lis)
        # fig,axes = plt.subplots(n,1)
        # maxtick = 0
        # for k in range(n):
        #     try :
        #         ax = axes[k]
        #     except :
        #         ax = axes
        #     arr = np.array(errs_lis[k])
        #     if (arr.shape[0]>maxtick):
        #         maxtick = arr.shape[0]
        #     ax.scatter(np.arange(0,arr.shape[0],1),arr,color='red')#,s=0.5)
        #     for ki in range(len(vertical_lines_xs[k])):
        #         ax.axvline(vertical_lines_xs[k][ki],color="blue")
        # for k in range(n):
        #     try :
        #         ax = axes[k]
        #     except :
        #         ax = axes
        #     ax.set_xlim([0,maxtick+1])
        # # plt.plot(xs,np.mean(arr,axis=0),color='red')
        # plt.show()
