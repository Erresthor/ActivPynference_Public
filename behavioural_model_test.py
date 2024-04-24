import pickle,os
import numpy as np
import matplotlib.pyplot as plt
import scipy

from demo_draft.my_projects.behavioural_task.grid import behavioural_process,basic_model,complex_grid_model,sub2ind,ind2sub

import actynf
from actynf.layer.layer_components import link_function
from actynf.architecture.network import network

def running_mean(x, N):
    return np.convolve(x, np.ones(N)/N, mode='same')

def clever_running_mean(arr, N):
    xarr = np.array(arr)
    xpost = np.zeros(xarr.shape)
    # raw_conv = np.convolve(x, np.ones(N)/N, mode='same')
    for k in range(xarr.shape[0]):
        localmean = 0.0
        cnt = 0.0
        for i in range(k-N,k+N+1,1):
            if ((i>=0) and (i<xarr.shape[0])):
                localmean += xarr[i]
                cnt += 1
        xpost[k] = localmean/(cnt+1e-18)
    return xpost

def generate_random_vector(N,rng):
    return np.array([rng.random() for k in range(N)])

def generate_random_array(N):
    return np.random.random((N,))

def training_curves_confidence_plot(list_of_confs,basepath=None):
    full_col_base = np.array([0,0,0,1.0])
    transp_col_base = np.array([0,0,0,0.5])

    K = None # Plot the first K timesteps, else leave as None

    fig,axes = plt.subplots(len(list_of_confs),sharex=True)
    fig.suptitle("Simulated subject behaviour for a feedback control task with ambiguous action modality")
    axes[-1].set_xlabel("Trials")
    for i,ax in enumerate(axes):
        ax.set_ylabel("Feedback")
        ax.set_title("Initial action confidence = " + str(list_of_confs[i]))
        ax.grid()
        ax.legend()

    for k,conf_value in enumerate(list_of_confs):
        # Attempt opening saved datafile           
        if not(actynf.isField(basepath)) :
            savepath = os.path.join("simulation_outputs","capped","1d",str(conf_value)+".wider_prefs.pckl")
        else : 
            savepath = basepath
        with open(savepath, 'rb') as handle:
            saved_data = pickle.load(handle)

        stms = saved_data['stms']
        weights = saved_data['weights']

        Nsubj = len(stms)
        Ntrials = len(stms[1]) # including the "0th" trial
        T = stms[0][1][0].o.shape[-1]
        xs = np.linspace(0,Ntrials-1,(Ntrials-1)*T)
        full_col_base = np.array([0,0,0,1.0])
        transp_col_base = np.array([0,0,0,0.5])
        
    
        for subj in range(Nsubj):
            print("------------------------------")
            subjd_stms = stms[subj]

            Fbs = []
            for trial in range(1,Ntrials): # Offset by 1
                trial_stms = subjd_stms[trial]
                process_recorded_obs = trial_stms[0].o[0,:]
                print(trial_stms[1].u)
                # print(trial_stms[1].u)
                for t in range(process_recorded_obs.shape[0]) :
                    Fbs.append(process_recorded_obs[t]/9.0)
            

            if (subj==Nsubj-1):
                col = np.zeros((3,))
            else :
                col = np.random.random((3,))

            full_col = np.copy(full_col_base)
            full_col[:3] = col
            transp_col = np.copy(transp_col_base)
            transp_col[:3] = col

            ax = axes[k]
            ax.plot(xs[:K],clever_running_mean(Fbs,10)[:K],color=full_col,linewidth=3)
            ax.scatter(xs[:K],Fbs[:K], s=1,color=transp_col)
            ax.plot(xs[:K],clever_running_mean(Fbs,2)[:K],color=transp_col,linewidth=0.1)
    fig.show()
    input()


def normal(x, mu, sigm):
    return (1.0/(sigm*np.sqrt(2.0*3.1415)))*np.exp(-0.5*np.power((x-mu)/sigm,2))

def plot_action(angle_action, position_action, distance_action,N=100):
    std_factor = 0.3
    def position_mapping(pos):
        mu_x = 1.0/6.0 + (1.0/3.0)*(pos//3)
        mu_y = 1.0/6.0 + (1.0/3.0)*(pos% 3)
        sigm = std_factor*1.0/3.0
        return [mu_x,sigm],[mu_y,sigm]
    
    def angle_mapping(ang):
        if (ang==0):
            return None,None
        mu_ang = 2*3.1514*((ang-1)/8)
        sigm_ang = std_factor*(2*3.1514/8.0)
        return mu_ang,sigm_ang

    def distance_mapping(dist):
        # Arbitrary correspondances :
        # A big distance is 30 to 100 % of the size of the diag
        # A medium distance is 15 to 30 % of the size of the diag
        # A small distance is 0 to 15 % of the size of the diag
        if dist==0:
            return 0.075,std_factor*0.075
        if dist==1:
            return 0.225,std_factor*0.075
        if dist==2:
            return 0.5,std_factor*0.2
    
    # Center coordinates : 
    theta_x,theta_y = position_mapping(position_action)
    draw_positions_x = np.random.normal(theta_x[0],theta_x[1],(N,))
    draw_positions_y = np.random.normal(theta_y[0],theta_y[1],(N,))
    center_point_coords = np.vstack([draw_positions_x,draw_positions_y])

    # Angle
    mu_ang,sigm_ang = angle_mapping(angle_action)
    if (actynf.isField(mu_ang)):
        draw_angle = np.random.normal(mu_ang,sigm_ang,(N,))
    else : 
        draw_angle = None
    
    mu_dist,sigm_dist = distance_mapping(distance_action)
    draw_dist = np.random.normal(mu_dist,sigm_dist,(N,))

    points = points_from_definite_coordinates(center_point_coords,draw_angle,draw_dist)
    return points
    # # PDF
    # arr = np.zeros((N,N))
    # xs = np.linspace(0,1,N)
    # ys = np.linspace(0,1,N)
    # for x_idx, x in np.ndenumerate(xs):
    #     for y_idx, y in np.ndenumerate(ys):
    #         theta_x,theta_y = position_mapping(position_action)
    #         arr[x_idx,y_idx] = normal(x,theta_x[0],theta_x[1])*normal(y,theta_y[0],theta_y[1])
    # plt.imshow(arr)
    # plt.show()

def points_from_definite_coordinates(pos,angle,distance):
    """ Returns the (x1,x2), (y1,y2) coordinates of the points
    from their 3 (quasi) orthogonal components. 
    """

    def rotate_vector(vector, angle):
        """ Vector and angle are array like"""
        x = vector[0] * np.cos(angle) - vector[1] * np.sin(angle)
        y = vector[0] * np.sin(angle) + vector[1] * np.cos(angle)
        return np.array([x,y])

    if not(actynf.isField(angle)):
        rotated_x1 = pos
        rotated_x2 = pos
        return rotated_x1,rotated_x2
    else :
        # Some values are incompatible ! 
        # if one of the two points gets outside the range of the
        # screen, what should we do ?
        unit_vector = np.vstack([0.5*distance,np.zeros(distance.shape)])
        unrotated_x1  = -unit_vector
        unrotated_x2  = unit_vector
        rotated_x1 = rotate_vector(unrotated_x1,angle) + pos
        rotated_x2 = rotate_vector(unrotated_x2,angle) + pos
        # print(rotated_x1,rotated_x2)
        for rotated_point in [rotated_x1,rotated_x2] :
            rotated_point[rotated_point>1.0] = 1.0
            rotated_point[rotated_point<0.0] = 0.0
        return rotated_x1,rotated_x2

def exploit_1d_model(generalize_temperature):
    Ntrials = 30
    Nsubj = 20
    n_feedback_ticks = 10
    T = 10
    Th = 2
    initial_action_mapping_confidence = 0.01
    override = False
    state_cap = 3
    action_cap = 2
    interpolation_model = actynf.LINEAR
    
    savepath = os.path.join("simulation_outputs","temp.pickle")
    # training_curves_confidence_plot([initial_action_mapping_confidence,1.0],basepath =  os.path.join("simulation_outputs","noncapped","1d"))
    
    # savepath = os.path.join("test_to_delete")
    if not(os.path.isfile(savepath)) or override:
        grid_size = (7,7)
        start_coords = [[5,1],[5,2],[4,1]]
        end_coords = [0,6]

        process = behavioural_process(grid_size,start_coords,end_coords,n_feedback_ticks,
            T,Th,seed=None)

        angle_model = basic_model(n_feedback_ticks,T,Th,"angle",initial_action_mapping_confidence,structure_hypothesis=interpolation_model,
                                state_cap = state_cap,action_cap = action_cap,generalize_temperature=generalize_temperature)
        print(angle_model)

        position_model = basic_model(n_feedback_ticks,T,Th,"position",initial_action_mapping_confidence,structure_hypothesis=interpolation_model,
                                     state_cap = state_cap,action_cap = action_cap,generalize_temperature=generalize_temperature)
        print(position_model)

        distance_model = basic_model(n_feedback_ticks,T,Th,"distance",initial_action_mapping_confidence,structure_hypothesis=interpolation_model,
                                     state_cap = state_cap,action_cap = action_cap,generalize_temperature=generalize_temperature)
        print(distance_model)

        angle_model.inputs.o = link_function(process, (lambda l: l.o))
        position_model.inputs.o = link_function(process, (lambda l: l.o))
        distance_model.inputs.o = link_function(process, (lambda l: l.o))

        process.inputs.u = link_function([angle_model,distance_model], (lambda l_ang,l_dist: (l_ang.u if (l_dist.u != 0) else np.array([0]))))

        behavioural_basic_network = network([process,angle_model,position_model,distance_model],"basic_network")
        
        
        networks = []
        stmlist = []
        weightlist = []
        for netidx in range(Nsubj):
            new_network = behavioural_basic_network.copy_network(netidx)
            networks.append(new_network)

            STMs,weights = new_network.run_N_trials(Ntrials,return_STMs=True,return_weights=True)
            stmlist.append(STMs)
            weightlist.append(weights)
        
        # for l in range(10):
        #     process_layer = behavioural_basic_network.layers[0]
        #     print(process_layer.STM.o)
        #     true_states = process_layer.STM.x
        #     for s in range(true_states.shape[0]):
        #         print(ind2sub(grid_size,true_states[s]))
        #     print()
        
        # Show the avg reward map for the experiment
        # -------------------------------------------
        # process_layer = behavioural_basic_network.layers[0]
        # feedback_arr = np.zeros(grid_size)
        # for s in range(process_layer.a[0].shape[1]):
        #     print(ind2sub(grid_size,s))
        #     x,y = ind2sub(grid_size,s)
        #     feedback_arr[x,y] = np.inner(process_layer.a[0][:,s],range(10))
        # import matplotlib.pyplot as plt
        # plt.imshow(feedback_arr, interpolation='none')
        # plt.show()
        # -------------------------------------------

        save_this = {
            "stms" : stmlist,
            "weights" : weightlist
        }

        with open(savepath, 'wb') as handle:
            pickle.dump(save_this, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    with open(savepath, 'rb') as handle:
        saved_data = pickle.load(handle)

    stms = saved_data['stms']
    weights = saved_data['weights']

    # Simulated transiton matrix : 
    Nsubj = len(stms)
    Ntrials = len(stms[0])-1
    xs = np.linspace(0,Ntrials,Ntrials*T)

    # print("Sanity check :")
    # print(np.round(weights[0][0][1]['b'][0][:,:,2]),2)
    # # 0th subj, 0th trial, angle model, action matrix, 0th factor
    
    n_plot_trial = 11
    n_plot_subj = 3
    fig,axes = plt.subplots(n_plot_subj,n_plot_trial-1)
    for subj in range(n_plot_subj):
        cnt = 0

        subjd_stms = stms[subj]
        subjd_weights = weights[subj]
        print("---")
        for trial in range(1,n_plot_trial):
            # print(len(subjd_weights))
            weights_trial = subjd_weights[trial]
            stms_trial = subjd_stms[trial]

            received_feedback = stms_trial[0].o
            ang_act = stms_trial[1].u
            print(ang_act)
            pos_act = stms_trial[2].u
            dist_act = stms_trial[3].u
            for t in range(T-1):
                # N = 1
                # print(received_feedback)
                if received_feedback[0,t] == 9:
                    break
                x,y = plot_action(ang_act[t],pos_act[t],dist_act[t],1)
                # print(x,y)
                
                axes[subj,trial-1].scatter(y[0,:],y[1,:],color="red",s=30)
                axes[subj,trial-1].scatter(x[0,:],x[1,:],color="blue",s=20)
                for k in range(x.shape[1]):
                    axes[subj,trial-1].plot([x[0,k],y[0,k]],[x[1,k],y[1,k]],color="black",zorder=-99)
                axes[subj,trial-1].set_xlim([1,0])
                axes[subj,trial-1].set_ylim([1,0])
                axes[subj,trial-1].set_aspect(1.0)
                axes[subj,trial-1].invert_xaxis()
                axes[subj,trial-1].invert_yaxis()

            
    fig.show()
    input()


    # N = 150
    # x,y = plot_action(6,2,0,N)
    # for k in range(x.shape[1]):
    #     plt.plot([x[0,k],y[0,k]],[x[1,k],y[1,k]],color="black")
    # plt.scatter(y[0,:],y[1,:],color="blue",s=30)
    # plt.scatter(x[0,:],x[1,:],color="red",s=20)

    if False:
        # Matrix learning plot
        plot_those_trials = [0,1,2,3,5,10,20,29]
        fig,axes = plt.subplots(len(plot_those_trials),Nsubj)
        for subj in range(Nsubj):
            
            cnt = 0

            subjd_stms = stms[subj]
            subjd_weights = weights[subj]
            for trial in range(Ntrials):
                # print(len(subjd_weights))
                weights_trial = subjd_weights[trial]
                # print(weights_trial[1])
                angle_model_weights = weights_trial[1]
                # print(weights_trial['b'])
                if (trial in plot_those_trials):
                    action_mat = actynf.normalize(angle_model_weights['b'][0][:,:,2])
                    axes[cnt,subj].imshow(action_mat)
                    cnt += 1
        fig.show()
        input()
    
    training_curves_confidence_plot([0.01,0.0],basepath = savepath)

def exploit_2D_model():
    Ntrials = 30
    Nsubj = 5
    n_feedback_ticks = 15
    T = 10
    Th = 2
    initial_action_mapping_confidence = 0.01
    override = False
    state_cap = 2
    action_cap = 3
    interpolation_model = actynf.LINEAR
    savepath = os.path.join("simulation_outputs","behavioural_task","noncapped","2d",str(initial_action_mapping_confidence)+"50trials_test_high_fb_res_multi_subj")
    # savepath = os.path.join("simulation_outputs","noncapped","2d",str(initial_action_mapping_confidence)+"test_initial_explo")
    # savepath = os.path.join("simulation_outputs","capped","2d",str(initial_action_mapping_confidence)+".wider_prefs.20ticks.20withprior.pckl")
    # training_curves_confidence_plot([initial_action_mapping_confidence,1.0],basepath =  os.path.join("simulation_outputs","noncapped","1d"))
    
    # savepath = os.path.join("test_to_delete")
    if not(os.path.isfile(savepath)) or override:
        grid_size = (7,7)
        start_coords = [[5,1],[5,2],[4,1]]
        end_coords = [0,6]

        process = behavioural_process(grid_size,start_coords,end_coords,n_feedback_ticks,
            T,Th,seed=None) # Nothing changes here

        grid_size = (7,7)
        angle_model = complex_grid_model(n_feedback_ticks,
                        grid_size,            
                        T,Th,"angle",
                        initial_action_mapping_confidence,
                        structure_hypothesis=interpolation_model,
                        state_cap = state_cap,action_cap = action_cap)
        print(angle_model)

        position_model = complex_grid_model(n_feedback_ticks,
                        grid_size,            
                        T,Th,"position",
                        initial_action_mapping_confidence,
                        structure_hypothesis=interpolation_model,
                        state_cap = state_cap,action_cap = action_cap)
        print(position_model)

        distance_model = complex_grid_model(n_feedback_ticks,
                        grid_size,            
                        T,Th,"distance",
                        initial_action_mapping_confidence,
                        structure_hypothesis=interpolation_model,
                        state_cap = state_cap,action_cap = action_cap)
        print(distance_model)

        angle_model.inputs.o = link_function(process, (lambda l: l.o))
        position_model.inputs.o = link_function(process, (lambda l: l.o))
        distance_model.inputs.o = link_function(process, (lambda l: l.o))

        # process.inputs.u = link_function([angle_model,distance_model], (lambda l_ang,l_dist: (l_ang.u if (l_dist.u != 0) else np.array([0]))))
        process.inputs.u = link_function([angle_model,distance_model], (lambda l_ang,l_dist: l_ang.u))

        behavioural_complex_network = network([process,angle_model,position_model,distance_model],"complex_network")
        
        
        networks = []
        stmlist = []
        weightlist = []
        for netidx in range(Nsubj):
            new_network = behavioural_complex_network.copy_network(netidx)
            networks.append(new_network)

            STMs,weights = new_network.run_N_trials(Ntrials,return_STMs=True,return_weights=True)
            stmlist.append(STMs)
            weightlist.append(weights)
        
        # Show the avg reward map for the experiment
        # -------------------------------------------
        # process_layer = behavioural_basic_network.layers[0]
        # feedback_arr = np.zeros(grid_size)
        # for s in range(process_layer.a[0].shape[1]):
        #     print(ind2sub(grid_size,s))
        #     x,y = ind2sub(grid_size,s)
        #     feedback_arr[x,y] = np.inner(process_layer.a[0][:,s],range(10))
        # import matplotlib.pyplot as plt
        # plt.imshow(feedback_arr, interpolation='none')
        # plt.show()
        # -------------------------------------------

        save_this = {
            "stms" : stmlist,
            "weights" : weightlist
        }
        with open(savepath, 'wb') as handle:
            pickle.dump(save_this, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    with open(savepath, 'rb') as handle:
        saved_data = pickle.load(handle)

    stms = saved_data['stms']
    weights = saved_data['weights']


    # # LEARNING CURVE
    fig,ax = plt.subplots()
    xs = np.linspace(0,Ntrials,Ntrials*T)
    for subj in range(Nsubj):
        color = np.random.random((3,))
        all_os = []
        for trial in range(1,Ntrials+1):
            print("----")
            print("FINAL")
            print(stms[subj][trial][0].u)
            print("LAYERS")
            print(stms[subj][trial][1].u)
            print(stms[subj][trial][3].u)
            print("----")
            for t in range(T): 
                o = stms[subj][trial][1].o[0,t]
                all_os.append(o)
            # axes[trial].imshow(np.outer(d_mat[0],d_mat[1]))
        ax.scatter(xs,all_os,s=1)
        ax.plot(xs,clever_running_mean(all_os,10),color=color,linewidth=2)
    ax.grid()
    fig.show()
    input()


    # Simulated transiton matrix : 
    # Nsubj = len(stms)
    # Ntrials = len(stms[0])-1
    # xs = np.linspace(0,Ntrials,Ntrials*T)

    # fig,axes = plt.subplots(Ntrials+1)
    # for trial in range(Ntrials+1):
    #     d_mat = weights[0][trial][1]["d"]
    #     axes[trial].imshow(np.outer(d_mat[0],d_mat[1]))
    # fig.show()
    # input()

    # EXPLORE LEARNT STATE MAPPINGS
    explore_these = [1,2,3,4,5,10,19,25,26,27,28,29]
    for trial in (explore_these):
        fig,axes = plt.subplots(3,T)
        for t in range(T):
            axes[0,t].imshow(np.sum(stms[0][trial+1][1].x_d[...,t], axis=(2,3)))
            axes[1,t].imshow(np.sum(stms[0][trial+1][1].x_d_smoothed[...,t], axis=(2,3)))
            axes[2,t].imshow(np.sum(stms[0][trial+1][1].x_d_smoothed[...,t], axis=(0,1)))
        fig.show()
    input()

    # EXPLORE LEARNT ACTION MAPPINGS 
    # explore_these = [1,2,3,4,5,10,19]
    # for trial in (explore_these):
    #     fig,axes = plt.subplots(3,1)
    #     for model_id in range(1,4):
    #         model = weights[0][trial][model_id]["b"]
    #         print(len(model),model[0].shape)
    #         for pi in range(model[2].shape[-1]):
    #             axes[model_id-1].imshow(actynf.normalize(model[2][...,pi]))
    #     fig.suptitle("FACTOR : 0")
    #     fig.show()
    # input()

    # EXPLORE LEARNT ACTION MAPPINGS
    # explore_these = [1,2,3,4,5,10,19]
    # for trial in (explore_these):
    #     fig,axes = plt.subplots(3,9)
    #     for model_id in range(1,4):
    #         model = weights[0][trial][model_id]["b"]
    #         print(len(model),model[0].shape)
    #         for pi in range(model[0].shape[-1]):
    #             axes[model_id-1,pi].imshow(actynf.normalize(model[0][...,pi]))
    #     fig.suptitle("FACTOR : 0")

    #     fig2,axes = plt.subplots(3,9)
    #     for model_id in range(1,4):
    #         model = weights[0][trial][model_id]["b"]
    #         print(len(model),model[0].shape)
    #         for pi in range(model[0].shape[-1]):
    #             axes[model_id-1,pi].imshow(actynf.normalize(model[1][...,pi]))
    #     fig2.suptitle("FACTOR : 1")
    #     fig.show()
    #     fig2.show()
    # input()

    # print("##########################################")
    # print(Nsubj)
    # for subj in range(Nsubj):
    #     # print(len(stms[subj]))
    #     for trial in range(1,Ntrials+1):
    #         print(np.round(stms[subj][trial][1].u_d,2))

    #         # print(np.round(stms[0][trial][1].u_d,2))
    # print("##########################################")


    # print("Sanity check :")
    # print(np.round(weights[0][0][1]['b'][0][:,:,2]),2)
    # # 0th subj, 0th trial, angle model, action matrix, 0th factor
    
    # n_plot_trial = 11
    # n_plot_subj = 3
    # fig,axes = plt.subplots(n_plot_subj,n_plot_trial-1)
    # for subj in range(n_plot_subj):
    #     cnt = 0

    #     subjd_stms = stms[subj]
    #     subjd_weights = weights[subj]
    #     print("---")
    #     for trial in range(1,n_plot_trial):
    #         # print(len(subjd_weights))
    #         weights_trial = subjd_weights[trial]
    #         stms_trial = subjd_stms[trial]

    #         received_feedback = stms_trial[0].o
    #         ang_act = stms_trial[1].u
    #         print(ang_act)
    #         pos_act = stms_trial[2].u
    #         dist_act = stms_trial[3].u
    #         for t in range(T-1):
    #             # N = 1
    #             # print(received_feedback)
    #             if received_feedback[0,t] == 9:
    #                 break
    #             x,y = plot_action(ang_act[t],pos_act[t],dist_act[t],1)
    #             # print(x,y)
                
    #             axes[subj,trial-1].scatter(y[0,:],y[1,:],color="red",s=30)
    #             axes[subj,trial-1].scatter(x[0,:],x[1,:],color="blue",s=20)
    #             for k in range(x.shape[1]):
    #                 axes[subj,trial-1].plot([x[0,k],y[0,k]],[x[1,k],y[1,k]],color="black",zorder=-99)
    #             axes[subj,trial-1].set_xlim([1,0])
    #             axes[subj,trial-1].set_ylim([1,0])
    #             axes[subj,trial-1].set_aspect(1.0)
    #             axes[subj,trial-1].invert_xaxis()
    #             axes[subj,trial-1].invert_yaxis()

            
    # fig.show()
    # input()


    # N = 150
    # x,y = plot_action(6,2,0,N)
    # for k in range(x.shape[1]):
    #     plt.plot([x[0,k],y[0,k]],[x[1,k],y[1,k]],color="black")
    # plt.scatter(y[0,:],y[1,:],color="blue",s=30)
    # plt.scatter(x[0,:],x[1,:],color="red",s=20)

    if False:
        # Matrix learning plot
        plot_those_trials = [0,1,2,3,5,10,20,29]
        fig,axes = plt.subplots(len(plot_those_trials),Nsubj)
        for subj in range(Nsubj):
            
            cnt = 0

            subjd_stms = stms[subj]
            subjd_weights = weights[subj]
            for trial in range(Ntrials):
                # print(len(subjd_weights))
                weights_trial = subjd_weights[trial]
                # print(weights_trial[1])
                angle_model_weights = weights_trial[1]
                # print(weights_trial['b'])
                if (trial in plot_those_trials):
                    action_mat = actynf.normalize(angle_model_weights['b'][0][:,:,2])
                    axes[cnt,subj].imshow(action_mat)
                    cnt += 1
        fig.show()
        input()
    
    # training_curves_confidence_plot([0.01,0.01],basepath = os.path.join("simulation_outputs","noncapped","1d","low_resolution"))

if __name__ == '__main__':
    exploit_1d_model(2.5)
    # exploit_2D_model()

