import numpy as np
import random 
import math
import matplotlib.pyplot as plt

SPRING_CST = 10
MAGNET_FORCE = 0.0005#0.005
# STATIC_LINK_FORCE = 1
LIMIT = 1
EQ_DIST = 10
FROT_CST = 0.5

class link :
    def __init__(self,from_n,to_n):
        self.from_node = from_n
        self.to_node = to_n

class node :
    def __init__(self):
        self.links = []
        self.node_idx = 0

    def linkto(self,other_node):
        self.links.append(link(self,other_node))
        other_node.links.append(link(self,other_node))

def get_connections(a_node):
    return_this = [[],[]]
    for link in a_node.links:
        if link.from_node==a_node:
            return_this[0].append(link.to_node)
        else :
            return_this[1].append(link.from_node)
    return return_this

def get_average_connections(nodelist):
    avg = 0
    for node in nodelist:
        avg += len(get_connections(node)[0])
    return avg/len(nodelist)

def dynamic_ordering(list_of_nodes,repulsive_force_max = 0,
                     max_iterations=100,delta_t = 0.1,randomstart=True):
    avg_conn_per_node = get_average_connections(list_of_nodes)
    print("Ordering network with an average of " + str(avg_conn_per_node) + " connections per node.")
    N = len(list_of_nodes)
    max_spring_force = SPRING_CST/N

    for nix in range(N):
        list_of_nodes[nix].node_idx = nix
    temps=[]
    z_pos = np.zeros((N,max_iterations))
    I = 3*EQ_DIST
    if (randomstart):
        z_pos[:,0] = (I*np.random.random((N,))-1.0) #Initial positions between -1 and 1
    dxs = np.zeros((N,max_iterations))
    ddxs = np.zeros((N,max_iterations))

    for it in range(1,max_iterations):
        for node_idx in range(N):           
            my_node = list_of_nodes[node_idx]
            my_zpos = z_pos[node_idx,it-1]
            connections = get_connections(my_node)
            
            frottements = -FROT_CST*dxs[node_idx,it-1]

            link_forces = 0
            for f_conn in connections[0] :
                # Force going from me to another node :
                # Equilibrium point is -EQ_DIST
                # Pulls me towards the bottom : (negative force)
                other_node_z = z_pos[f_conn.node_idx,it-1]
                link_forces += -max_spring_force*(EQ_DIST + (my_zpos-other_node_z))
                # If my_zpos = other_node_z - EQ_DIST --> F = 0
                # If my_zpos < other_node_z - EQ_DIST --> F > 0
            for b_conn in connections[1] : 
                # Force going from another node to me :
                # Pulls me towards the top (positive force)
                other_node_z = z_pos[b_conn.node_idx,it-1]
                link_forces += -max_spring_force*(-EQ_DIST + (my_zpos-other_node_z))
                # If my_zpos = other_node_z + EQ_DIST --> F = 0
                # If my_zpos < other_node_z + EQ_DIST --> F > 0

            repulsive_force = 0
            if (repulsive_force_max > 1e-8) :
                for other_nodes_idx in range(N):
                    if (other_nodes_idx != node_idx):
                        other_node =list_of_nodes[other_nodes_idx]
                        other_nodes_zpos = z_pos[other_nodes_idx,it-1]
                        potential_add = MAGNET_FORCE/(my_zpos - other_nodes_zpos)
                        if (abs(potential_add)>repulsive_force_max) :
                            potential_add = math.copysign(repulsive_force_max,potential_add)
                            # Force the nodes to be distanced from each other
                        repulsive_force +=potential_add

            ddxs[node_idx,it] = (repulsive_force + link_forces + frottements)
            # For each node, calculate the sum of forces applied to it :
            dxs[node_idx,it] = dxs[node_idx,it-1] + ddxs[node_idx,it]*delta_t
            z_pos[node_idx,it] = z_pos[node_idx,it-1] + dxs[node_idx,it]*delta_t
    return z_pos,dxs,ddxs,temps

def mechanical_equations_ordering(list_of_neighbors,max_iter = 100,repulsive_cst = 0,opt=0):
    if opt==0:
        xs,dxs,ddxs = spring_forces_trajectories(list_of_neighbors,repulsive_force_max = repulsive_cst,max_iterations=max_iter)
    elif opt==1:
        xs,dxs,ddxs = repulsive_forces_trajectories(list_of_neighbors,repulsive_force_max = repulsive_cst,max_iterations=max_iter)
    return xs,dxs,ddxs

def spring_forces_trajectories(list_of_neighbors,repulsive_force_max = 0,
                     max_iterations=100,delta_t = 0.1,randomstart=True):
    N = len(list_of_neighbors)
    max_spring_force = SPRING_CST/N

    z_pos = np.zeros((N,max_iterations))
    if (randomstart):
        I = EQ_DIST # Initial random position spread
        z_pos[:,0] = (I*(np.random.random((N,))-1.0)) #Initial positions between -1 and 1
    dxs = np.zeros((N,max_iterations))
    ddxs = np.zeros((N,max_iterations))

    for it in range(1,max_iterations):
        for node_idx in range(N):           
            my_zpos = z_pos[node_idx,it-1]
            connections = list_of_neighbors[node_idx]
            
            frottements = -FROT_CST*dxs[node_idx,it-1]

            link_forces = 0
            for b_conn in connections[0] : # UPSTREAM FORCES
                # Force going from another node to me :
                # Pulls me towards the top (positive force)
                other_node_z = z_pos[b_conn[0],it-1]
                link_forces += -max_spring_force*(-EQ_DIST + (my_zpos-other_node_z))*b_conn[1]
                # If my_zpos = other_node_z + EQ_DIST --> F = 0
                # If my_zpos < other_node_z + EQ_DIST --> F > 0

            for f_conn in connections[1] : # DOWNSTREAM FORCES
                # Force going from me to another node :
                # Equilibrium point is -EQ_DIST
                # Pulls me towards the bottom : (negative force)
                other_node_z = z_pos[f_conn[0],it-1]
                link_forces += -max_spring_force*(EQ_DIST + (my_zpos-other_node_z))*f_conn[1]
                # If my_zpos = other_node_z - EQ_DIST --> F = 0
                # If my_zpos < other_node_z - EQ_DIST --> F > 0
            

            repulsive_force = 0
            if (repulsive_force_max > 1e-8) :
                for other_node_idx in range(N):
                    if (other_node_idx != node_idx):
                        other_nodes_zpos = z_pos[other_node_idx,it-1]
                        potential_add = MAGNET_FORCE/(my_zpos - other_nodes_zpos)
                        if (abs(potential_add)>repulsive_force_max) :
                            potential_add = math.copysign(repulsive_force_max,potential_add)
                            # Force the nodes to be distanced from each other
                        repulsive_force +=potential_add

            ddxs[node_idx,it] = (repulsive_force + link_forces + frottements)
            # For each node, calculate the sum of forces applied to it :
            dxs[node_idx,it] = dxs[node_idx,it-1] + ddxs[node_idx,it]*delta_t
            z_pos[node_idx,it] = z_pos[node_idx,it-1] + dxs[node_idx,it]*delta_t
    return z_pos,dxs,ddxs

def repulsive_forces_trajectories(list_of_neighbors,repulsive_force_max = 0,
                     max_iterations=100,delta_t = 0.1,randomstart=True):
    # If two connected nodes are within EQ_DIST,
    # then the exert a force towards the direction of the
    # connection proportionnal to the linear distance 
    # between the two
    
    N = len(list_of_neighbors)
    max_spring_force = SPRING_CST/N

    z_pos = np.zeros((N,max_iterations))
    if (randomstart):
        I = EQ_DIST # Initial random position spread
        z_pos[:,0] = (I*(np.random.random((N,))-1.0)) #Initial positions between -1 and 1
    dxs = np.zeros((N,max_iterations))
    ddxs = np.zeros((N,max_iterations))

    for it in range(1,max_iterations):
        for node_idx in range(N):           
            my_zpos = z_pos[node_idx,it-1]
            connections = list_of_neighbors[node_idx]
            
            frottements = -FROT_CST*dxs[node_idx,it-1]

            link_forces = 0
            for b_conn in connections[0] : # UPSTREAM FORCES
                # Force going from another node to me :
                # Pulls me towards the top (positive force)
                other_node_z = z_pos[b_conn[0],it-1]
                spring_force = -max_spring_force*(-EQ_DIST + (my_zpos-other_node_z))*b_conn[1]
                if (spring_force < 0):
                    spring_force = 0
                link_forces +=spring_force
                # If my_zpos = other_node_z + EQ_DIST --> F = 0
                # If my_zpos < other_node_z + EQ_DIST --> F > 0

            for f_conn in connections[1] : # DOWNSTREAM FORCES
                # Force going from me to another node :
                # Equilibrium point is -EQ_DIST
                # Pulls me towards the bottom : (negative force)
                other_node_z = z_pos[f_conn[0],it-1]
                spring_force = -max_spring_force*(EQ_DIST + (my_zpos-other_node_z))*f_conn[1]
                if (spring_force > 0):
                    spring_force = 0
                link_forces +=spring_force
                # If my_zpos = other_node_z - EQ_DIST --> F = 0
                # If my_zpos < other_node_z - EQ_DIST --> F > 0
            
            number_of_connections = len(connections[1]) + len(connections[0])

            repulsive_force = 0
            if (repulsive_force_max > 1e-8) :
                for other_node_idx in range(N):
                    if (other_node_idx != node_idx):
                        other_nodes_zpos = z_pos[other_node_idx,it-1]
                        potential_add = MAGNET_FORCE/(my_zpos - other_nodes_zpos)
                        if (abs(potential_add)>repulsive_force_max) :
                            potential_add = math.copysign(repulsive_force_max,potential_add)
                            # Force the nodes to be distanced from each other
                        repulsive_force +=potential_add

            # I guess mass could be the number of connections ? or size of the network (nbr of parameters) ?
            ddxs[node_idx,it] = (repulsive_force + link_forces + frottements) #/number_of_connections

            # For each node, calculate the sum of forces applied to it :
            dxs[node_idx,it] = dxs[node_idx,it-1] + ddxs[node_idx,it]*delta_t
            z_pos[node_idx,it] = z_pos[node_idx,it-1] + dxs[node_idx,it]*delta_t
    return z_pos,dxs,ddxs

def window_mean(arr, window_size):
    window = np.ones(window_size).astype(np.float) / window_size
    return np.convolve(arr, window, mode='same')

def plot_final(myarray,N_categories = 100):
    N_data = myarray.shape[0]
    fig,axs = plt.subplots(3)
    ax = axs[0]
    last25 = int(myarray.shape[1]*0.75)
    for k in range(N_data):
        ax.axvline(np.mean(myarray[k,last25:]))

    mean_arr = np.mean(myarray[:,last25:],axis=1)

    NP = 10

    sorted_arr = np.sort(mean_arr)
    maxi = np.max(sorted_arr)
    mini = np.min(sorted_arr)
    dist = (maxi-mini)
    cats = []
    counter_arr = np.zeros((N_categories+NP,))

    cutter_min = mini
    cutter_max = mini + (dist/N_categories)
    for n in range(N_categories+NP):
        cats.append((cutter_min+cutter_max)/2)
        for k in range(sorted_arr.shape[0]):
            if (sorted_arr[k]>=cutter_min) and (sorted_arr[k]<cutter_max):
                counter_arr[n] = counter_arr[n] + 1
        cutter_min = cutter_max
        cutter_max = cutter_max + (dist/N_categories) 

    flattened_dens = window_mean(counter_arr,2)
    # If the current measured by all the eeg comes from all neurons simultaneously
    mean_V = (flattened_dens/N_data)*0.04 + ((N_data-flattened_dens)/N_data)*(-0.07)
    deltaMicroV_mean = (mean_V + 0.07)*1e6
    ax2 = axs[1]
    
    print()
    ax2.scatter(cats,flattened_dens)

    ax3 = axs[2]
    ax3.plot(cats,deltaMicroV_mean)

def basic_leaky_plot():
    u_rest = -0.070
    u_back = -0.075
    u_max = -0.05
    u_spike = 0.02
    R = 4*10e6    # 4 MOhms
    T = 100000     # Total number of timesteps
    Tmin = 0       # 0 ms
    Tmax = 0.5     # 500 ms
    tau_m = 0.005   # 5 ms

    delta_T = (Tmax - Tmin)/T
    print(delta_T)
    ts = np.linspace(Tmin,Tmax,T)
    input_intensity = np.zeros((T,))
    for t_idx in range(T-1) :
        if (t_idx % int(T/10)) == 0 :
            input_intensity[t_idx:t_idx+int(T/10)] = 3*random.random()*1e-9 # Between 0 and 3 nA
    
    # input_intensity[int(T/2):] = 2.4*1e-9
    ut = np.zeros((T,))
    ut[0] = u_rest
    dut = np.zeros((T,))
    skip_next=False
    for t_idx in range(T-1) :           

        if (skip_next):
            skip_next = False
        else :
            print(R*input_intensity[t_idx])
            new_dut = (R*input_intensity[t_idx] - (ut[t_idx] - u_rest))/tau_m
            dut[t_idx+1] = new_dut
            ut[t_idx+1] = ut[t_idx] + dut[t_idx+1]*delta_T
            
            if ut[t_idx+1]>u_max :
                skip_next = True
                ut[t_idx+1] = u_spike
                dut[t_idx+1] = 0
                try :
                    ut[t_idx+2] = u_back
                    dut[t_idx+2] = 0
                except :
                    print("End reached !")
    fig,axs = plt.subplots(2)
    axs[0].plot(ts,ut)
    axs[1].plot(ts,input_intensity)
    plt.show()


def generate_messy_connections(N):
    nodeN = 600
    maxForwardConn = 50
    maxBackwardConn = 0
    connect_only_forwards = True
    L = []
    for k in range(nodeN):
        L.append([[],[]])
    for k in range(nodeN):
        number_of_forward_conn = random.randint(0,maxForwardConn)
        node_forward_connections = []
        for i in range(number_of_forward_conn):
            if connect_only_forwards :
                if k<nodeN-1:
                    connect_to = random.randint(k+1,nodeN-1)
                else :
                    continue
            else :
                connect_to = random.randint(0,nodeN-1)
            if (connect_to != k):
                node_forward_connections.append([connect_to,1])
                L[connect_to][1].append([k,1])

        number_of_backward_conn = random.randint(0,maxBackwardConn)
        node_backward_connections = []
        for i in range(number_of_backward_conn):
            connect_to = random.randint(0,nodeN-1)
            if (connect_to != k):
                node_backward_connections.append([connect_to,1])
                L[connect_to][0].append([k,1])

        L[k][0] += node_forward_connections
        L[k][1] += node_backward_connections
    return L

def get_x_trajectory_for_layered_net(nodeN,max_conn_per_node_within_lay,max_conn_per_node_between_lay,
                    MI = 500,opt=1):
    def get_1d_coord(lay_id, node_id):
        coord = 0
        for i in range(lay_id) :
            coord = coord + nodeN[i]
        coord = coord + node_id
        return coord

    list_of_nodes = []
    for layer_ix in range(len(nodeN)):
        for node_ix in range(nodeN[layer_ix]):
            list_of_nodes.append([[],[]])
    
    print(len(list_of_nodes))
    for layer_ix in range(len(nodeN)):
        for node_ix in range(nodeN[layer_ix]):
            list_of_node_idx = get_1d_coord(layer_ix,node_ix)
            # print(list_of_node_idx)

            number_of_conn_same_lay = random.randint(1,max_conn_per_node_within_lay[layer_ix])
            number_of_conn_next_lay = random.randint(0,max_conn_per_node_between_lay[layer_ix])
            
            slc_list = []
            for slc in range(number_of_conn_same_lay):
                connect_to_lay = layer_ix
                connect_to_node = random.randint(0,nodeN[connect_to_lay]-1)
                if (connect_to_node != node_ix):
                    other_list_of_node_idx = get_1d_coord(connect_to_lay,connect_to_node)
                    # print(other_list_of_node_idx,connect_to_lay,connect_to_node)
                    # print(list_of_nodes[other_list_of_node_idx][1])
                    slc_list.append([other_list_of_node_idx,1])
                    list_of_nodes[other_list_of_node_idx][0].append([list_of_node_idx,1]) # Backward connection
            
            blc_list = []
            for blc in range(number_of_conn_next_lay):
                connect_to_lay = layer_ix+1
                if (connect_to_lay<len(nodeN)):
                    connect_to_node = random.randint(0,nodeN[connect_to_lay]-1)
                    other_list_of_node_idx = get_1d_coord(connect_to_lay,connect_to_node)
                    # print(other_list_of_node_idx,connect_to_lay,connect_to_node)
                    # print(list_of_nodes[other_list_of_node_idx][1])
                    blc_list.append([other_list_of_node_idx,1])
                    list_of_nodes[other_list_of_node_idx][0].append([list_of_node_idx,1]) # Backward connection

            list_of_nodes[list_of_node_idx][1] += slc_list + blc_list
    # print(list_of_nodes)
    ts = np.arange(0,MI,1)

    x,xx,xxx = mechanical_equations_ordering(list_of_nodes,max_iter = MI,repulsive_cst = 0,opt=opt)
    return ts,x,xx,xxx

if __name__ == "__main__":
    # Generate a random number of nodes & links
    # # Chaotic layer
    # nodeN = 600
    # maxForwardConn = 50
    # maxBackwardConn = 0
    # connect_only_forwards = True
    # L = []
    # for k in range(nodeN):
    #     L.append([[],[]])
    # for k in range(nodeN):
    #     number_of_forward_conn = random.randint(0,maxForwardConn)
    #     node_forward_connections = []
    #     for i in range(number_of_forward_conn):
    #         if connect_only_forwards :
    #             if k<nodeN-1:
    #                 connect_to = random.randint(k+1,nodeN-1)
    #             else :
    #                 continue
    #         else :
    #             connect_to = random.randint(0,nodeN-1)
    #         if (connect_to != k):
    #             node_forward_connections.append([connect_to,1])
    #             L[connect_to][1].append([k,1])

    #     number_of_backward_conn = random.randint(0,maxBackwardConn)
    #     node_backward_connections = []
    #     for i in range(number_of_backward_conn):
    #         connect_to = random.randint(0,nodeN-1)
    #         if (connect_to != k):
    #             node_backward_connections.append([connect_to,1])
    #             L[connect_to][0].append([k,1])

    #     L[k][0] += node_forward_connections
    #     L[k][1] += node_backward_connections
    # print(L)
    # V1, V2, V4 layers ?
    nodeN = [500,500,2000]
    max_conn_per_node_within_lay = [1,3,5]
    max_conn_per_node_between_lay = [30,15,0]
    ts,x,useless,useless = get_x_trajectory_for_layered_net(nodeN,max_conn_per_node_within_lay,max_conn_per_node_between_lay)
    plot_final(x)

    nodeN = [500,500,2000]
    max_conn_per_node_within_lay = [1,3,5]
    max_conn_per_node_between_lay = [3,1,0]
    ts,x,useless,useless = get_x_trajectory_for_layered_net(nodeN,max_conn_per_node_within_lay,max_conn_per_node_between_lay)
    plot_final(x)
    plt.show()
    # ts = np.linspace(0,10,1000)
    # def action_potential(t,Vrest,DeltaV,t0,tau) :
    #     return Vrest + DeltaV *(1 - np.exp(-(t - t0) / tau))

    # axes[0].plot(ts,action_potential(ts,-0.070,0.1,ts[0],0.1))
    # plt.show()

    # for nod_idx in range(nodeN):
    #     nod = L[nod_idx]
    #     number_of_conn = random.randint(1,maxForwardConn)
    #     for i in range(number_of_conn):
    #         connect_to = random.randint(0,nodeN-1)
    #         if (connect_to != nod_idx):
    #             new_link = link(nod,L[connect_to])
    #             nod.links.append(new_link)
    #             L[connect_to].links.append(new_link)
    # # lower_node = node()
    # # higher_node =node()
    # # mid_1 = node()
    # # mid_2 = node()
    # # mid_3 = node()
    # # mid_4 = node()
    # # high2 = node()

    # # lower_node.linkto(mid_1)
    # # lower_node.linkto(mid_2)
    # # lower_node.linkto(mid_3)
    # # lower_node.linkto(mid_4)
    # # mid_1.linkto(higher_node)
    # # mid_2.linkto(higher_node)
    # # mid_3.linkto(higher_node)
    # # mid_4.linkto(high2)

    # # L = [lower_node,higher_node,mid_1,mid_2,mid_3,mid_4,high2]
    # nodeN = len(L)
    # cnt = 0
    # fig,axes = plt.subplots(2,2)
    # avg_x_final = np.zeros((nodeN,))
    # for ax in axes :
    #     for ax2 in ax:
    #         cnt +=1
    #         xs,dxs,ddxs,temps = dynamic_ordering(L,max_iterations=500)
    #         K = xs.shape[1]
    #         its = np.arange(0,K,1)
    #         avg_x_final = avg_x_final + xs[:,-1]
    #         for k in range(nodeN):
    #             ax2.plot(its,xs[k,:])
    # print(avg_x_final/cnt)
    # plt.show()