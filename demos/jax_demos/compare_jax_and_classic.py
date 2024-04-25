import numpy as np

import actynf
from utils.basic_task import build_training_process,build_subject_model


if __name__=='__main__':
    # Environment variables
    Ns = 5
    
    p_up = 0.8
    p_low = 0.3
    
    kas = [0.8,0.5]
    
    # Subject variables
    la = 2
    rs = 3
    T_horizon = 2
    initial_clue_confidence = 0.1
    action_selection_temperature = 32
    mem_loss=0.0
    
    A,B,D,U = build_training_process(Ns,p_up,p_low,kas)
    # print(B)
    
    
    kas_subj = [0.99,0.3]
    kb_subj = 0.1
    b_str = 1.0
    kd = 0.2
    a,b,c,d,e,u = build_subject_model(Ns,
                        kas_subj,
                        B,kb_subj,b_str,
                        U,
                        kd,
                        rs)
    
    
    # Classical actynf network building :
    T = 10
    Th = 2
    process_layer = actynf.layer("process","process",
                 A,B,None,D,None,
                 U,T,Th)
    
    model_layer = actynf.layer("model","model",
                 a,b,c,d,e,
                 U,T,Th)