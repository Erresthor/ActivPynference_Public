# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15th 2022

Use a genetic based algorithm to fit an [observations,observed_actions] dataset to an active inference paradigm

We take a random set of input parameters for N agents & run the simulation using those parameters (a,b,c,d,e). Alone, they will allow us to simulate a various range of behaviour.

We then compare the (T-1) action / (T) observations to the ones we wish to fit by calculating the difference between the various datasets. The p*N (0<p<1) ones with the best scores are kept.

We apply a small scale mutation to those elements and reproduce them. We proceeed as long as the overall distance is too elevated.

---> At some point, aim to find a gradient descent alternative ? (the stochastic nature of the paradigm might prove problematic)




Work in progress ^^"


@author: cjsan
"""
import numpy as np
import random
from PIL import Image, ImageDraw
import sys,os,inspect
import math
import matplotlib.pyplot as plt
import itertools
import random

from ..base.miscellaneous_toolbox import flexible_copy , isField
from ..base.function_toolbox import normalize
from ..base.function_toolbox import spm_dot,spm_kron
from ..base.plotting_toolbox import multi_matrix_plot
from ..base.file_toolbox import load_flexible,save_flexible
from ..layer_old.mdp_layer import mdp_layer
from ..layer_old.layer_postrun import evaluate_run

def initialize_layer(T,a,b,c,d,e,u,o):
    layer = mdp_layer()
    layer.options.T_horizon = 1
    layer.T = T

    layer.A_ = a
    layer.a_ = a

    layer.B_ = b
    layer.b_ = b

    layer.C_ = c

    layer.d_ = d
    layer.D_ = d


    layer.o = o
    #print(o)
    # E / e not implemented yet :D


    # Action based : let us allow any combination of action for now
    layer.U_ = u

    return layer

def generate_random_layer(layer_dims,layer) :
    Ns = layer_dims[0]
    No = layer_dims[1]
    Nu = layer_dims[2]
    T = layer_dims[3]
    u = layer_dims[4]
    o = layer_dims[5]

    a = []
    for mod in range(len(No)):
        a.append(normalize(np.random.random((No[mod],)+tuple(Ns))))
    
    b = []
    for fac in range(len(Ns)):
        b.append(normalize(np.random.random((Ns[fac],Ns[fac],Nu[fac]))))
        b[-1]=(layer.b_[fac]) 

    c = []
    for mod in range(len(No)):
        #c.append(normalize(np.random.random((No[mod],1))))
        c.append(layer.C_[mod])
    
    d = [] 
    for fac in range(len(Ns)):
        d.append(normalize(np.random.random((Ns[fac]))))

    return initialize_layer(T, a, b, c, d, None, u,o)

def initialize_population(simulation_size_tuple, layer_dims,layer):
    pop = []
    N = simulation_size_tuple[0]
    n = simulation_size_tuple[1]
    for sub in range(N):
        pop.append([generate_random_layer(layer_dims,layer)])
        for random_instance in range(1,n):
            pop[sub].append(pop[sub][0].copy())
    return pop

def run_population(pop,train_afterwards = False):
    for sub in range(len(pop)):
        for inst in range(len(pop[sub])):
            print(str(sub*len(pop[0]) + inst + 1) + " / " + str(len(pop)*len(pop[0])),end='\r')
            pop[sub][inst].run(learn=train_afterwards)
            #print(pop[sub][inst].o)
    print()


def w_loss(layer,data):
    layer_obs = layer.O
    layer_acts = layer.u

    data_obs = data[0]
    data_acts = data[1]

    counter = 0.0
    total = 0.0
    for fac in range(layer_acts.shape[0]):
        for t in range(layer_acts.shape[1]):
            if (abs(layer_acts[fac,t]-data_acts[fac,t])>1e-5):
                total = total + 1.0
            counter = counter + 1
    acts_mse = total/counter
    return acts_mse

def sequence_loss(layer,data):
    layer_obs = layer.O
    layer_acts = layer.u

    data_obs = data[0]
    data_acts = data[1]

    counter = 0.0
    total = 0.0
    for fac in range(layer_acts.shape[0]):
        for t in range(layer_acts.shape[1]):
            if (abs(layer_acts[fac,t]-data_acts[fac,t])<1e-5):
                total = 0
            else :
                return ((layer_acts.shape[1] - t)/float(layer_acts.shape[1]))
            counter = counter + 1
    acts_mse = total/counter
    return acts_mse

def evaluate_pop(pop,data):
    errors = []
    for k in range(len(pop)):
        errors.append([])
        for l in range(len(pop[0])) :
            error_term = round(w_loss(pop[k][l], data),2)
            #error_term = round(sequence_loss(pop[k][l], data),2)
            errors[k].append(error_term)
            print(str(pop[k][l].u) + "  -  " + str(data[1]) + "  : err = " + str(errors[k]))   
            #print(pop[k][l].a_) 
    return errors

def pick_best_mutation_candidates(p,pop,error_estimates):
    # p is the part of the population mutated (0< <1)
    N = len(pop)
    n = len(pop[0])

    reduced_error = []
    for k in range(N):
        #for l in range(n):
        reduced_error.append(sum(error_estimates[k])/len(error_estimates[k]))
    sorted_indices = np.argsort(np.array(reduced_error))

    mutation_candidates = []
    corresponding_perf = []
    for mutated in range(int(p*N)) :
        mutation_candidates.append(pop[sorted_indices[mutated]][0])
        corresponding_perf.append(reduced_error[sorted_indices[mutated]])
    return mutation_candidates,corresponding_perf


def mutate_probability_density(prob_density,intens = 1) :
    mean = np.mean(prob_density)
    variance = np.var(prob_density)
    prob_density = prob_density + 2*intens*(np.random.random(prob_density.shape)-1)
    return prob_density

def mutate_matrix(matrix,prob,intens):
    if (type(matrix)==list):
        returner = []
        for mat in range(len(matrix)):
            returner.append(mutate_matrix(matrix[mat],prob,intens))
        return returner
    else :
        if (matrix.ndim ==1) :
            isMutating = (random.random() <= prob)
            if (isMutating):
                matrix = mutate_probability_density(matrix,intens)
            return matrix
        else :
            # We must decide weither or not a given column will mutate :
            dims = matrix.shape[1:]

            # it = np.nditer(matrix[0,...],flags=['multi_index'])
            # while not(it.finished):
            #     # probabilistic test here
            #     isMutating = (random.random() <= prob)
            #     if (isMutating):
            #         #print("MUTATION")
            #         tuple_prob_dist = [slice(None)]+ list(it.multi_index)
            #         tuple_prob_dist = (tuple(tuple_prob_dist))
            #         matrix[tuple_prob_dist] = mutate_probability_density( matrix[tuple_prob_dist],intens)
            #         #prob_dist = matrix[]
            #     it.iternext()
            it = np.nditer(matrix,flags=['multi_index'])
            while not(it.finished):
                # probabilistic test here
                isMutating = (random.random() <= prob)
                if (isMutating):
                    #print("MUTATION")
                    tuple_prob_dist = (it.multi_index)
                    tuple_prob_dist = tuple(tuple_prob_dist)
                    matrix[tuple_prob_dist] = matrix[tuple_prob_dist] + 2*intens*(random.random()-1)
                    if (matrix[tuple_prob_dist]<=0):
                        matrix[tuple_prob_dist] = 1e-3
                    #prob_dist = matrix[]
                it.iternext()

            
            return normalize(matrix)


def mutate(layer,mutation_intensity,mutation_prop):
    layer.a_ = mutate_matrix(layer.a_, mutation_prop, mutation_intensity)
    # layer.b_ = mutate_matrix(layer.b_, mutation_prop, mutation_intensity)
    layer.d_ = mutate_matrix(layer.d_, mutation_prop, mutation_intensity)

    return layer



def create_new_generation(shape_tuple,mutation_candidates,mutation_intensity,mutation_prob,layer_dims,layer):
    N = shape_tuple[0]
    n = shape_tuple[1]
    c = len(mutation_candidates)
    pimm = 0.1 # Proportion of new random immigrants


    # We keep the winners of the previous round : 
    new_generation = []
    for cand in range(c):
        new_generation.append([])
        for k in range(n):
            new_generation[cand].append(mutation_candidates[cand].copy())
    
    # We introduce new randomized weights 
    for immigrant in range(int(pimm*N)):
        new_generation.append([])
        random_sub = generate_random_layer(layer_dims,layer)
        for k in range(n):
            new_generation[-1].append(random_sub.copy())

    # And introduce mutated versions of the winners :
    candidate_counter = 0
    while (len(new_generation)<N):
        candidate_mutated_child = mutate(mutation_candidates[candidate_counter].copy(),mutation_prob,mutation_intensity)
        #print(candidate_mutated_child.a_)
        L = []
        for k in range(n):
            L.append(candidate_mutated_child.copy())
            L[-1].o = layer.o
            print(L[-1].o)
            print(L[-1].s)
        new_generation.append(L)
        candidate_counter = candidate_counter + 1
        if (candidate_counter>=c) :
            candidate_counter = 0
    return new_generation


def genetic_fit(ngen,layer,effective_shape,candidate_prop,weighlist = None):

    u = layer.U_
    o = layer.o
    layer_dims = [[5], [5], [3], 10, u,o]

    pop = initialize_population(effective_shape,layer_dims,layer)
    train_pop = False

    data = [layer.o,layer.u]

    new_gen = None
    for gen in range(ngen) :
        
        if (new_gen != None):
            pop = new_gen

        if(not(isField(weighlist))) :
            mutation_intensity = 0.5
        else : 
            mutation_intensity = weighlist[gen]
            
        print("--------------- GENERATION " + str(gen+1) +" --------------")
        # print(len(pop))
        # for k in range(len(pop)):
        #     print(len(pop[k]))
        run_population(pop,train_pop)

        L = evaluate_pop(pop, data)

        mutation_candidates,perf = pick_best_mutation_candidates(candidate_prop,pop, L)

        new_gen = create_new_generation(effective_shape, mutation_candidates, mutation_intensity,mutation_prob,layer_dims,layer)

        print(perf)
        print(mutation_candidates[0].a_)
    return mutation_candidates



def genetic_fit_dyn(ngen,layer_w,data,effective_shape,candidate_prop,mut_int= 0.5,mut_prob=0.5):

    u = layer_w.U_
    o = layer_w.o
    layer_dims = [[5], [5], [3], layer_w.T, u,o]

    new_gen = initialize_population(effective_shape,layer_dims,layer_w)
    train_pop = False

    old_perf = [1.0]
    new_perf = [1.0]
    
    counter = 0
    for gen in range(ngen) :
        print("--------------- GENERATION " + str(gen+1) +" --------------")
        print(counter)
        if (new_perf[0]-old_perf[0] >= -1e-3) :
            counter = counter + 1
            new_perf=old_perf
        else :
            counter = 0
        print(counter)

        if (counter >= 5):
            mut_int = mut_int/2.0
            #mut_prob = mut_prob/2.0
            counter = 0
            print("REDUCING INTENSITY")
        # print(len(pop))
        # for k in range(len(pop)):
        #     print(len(pop[k]))
        run_population(new_gen,train_pop)

        L = evaluate_pop(new_gen, data)
        
        old_perf = new_perf
        mutation_candidates,new_perf = pick_best_mutation_candidates(candidate_prop,new_gen, L)

        new_gen = create_new_generation(effective_shape, mutation_candidates, mut_int,mut_prob,layer_dims,layer_w)

        print(np.round(mutation_candidates[0].a_,2))
        #print(mutation_candidates[0].b_)    
        print(new_perf)
        
    return mutation_candidates






