function MDP = DEM_demo_MDP_XX
% Demo of active inference (T-maze): belief propagation scheme
%__________________________________________________________________________
%
% This routine uses a Markov decision process formulation of active
% inference (with belief propagation) to model foraging for information in
% a three arm maze.  This demo illustrates active inference in the context
% of Markov decision processes, where the agent is equipped with prior
% beliefs that it will minimise expected free energy in the future. This
% free energy is the free energy of future sensory states expected under
% the posterior predictive distribution. It can be regarded as a
% generalisation of KL control that incorporates information gain or
% epistemic value.
%
% In this example, the agent starts at the centre of a three way maze which
% is baited with a reward in one of the two upper arms. However, the
% rewarded arm changes from trial to trial.  Crucially, the agent can
% identify where the reward (US) is located by accessing a cue (CS) in the
% lower arm. This tells the agent whether the reward is on the left or the
% right upper arm.  This means the optimal policy would first involve
% maximising information gain or epistemic value by moving to the lower arm
% and then claiming the reward this signified. Here, there are eight hidden
% states (four locations times right or left reward), four control states
% (that take the agent to the four locations) and four exteroceptive
% outcomes (that depend on the agents locations) plus three interoceptive
% outcomes indicating reward (or not).
%
% This version focuses on a sophisticated AI implementation that replaces
% policies (i.e., ordered sequences of actions) with a deep tree search
% over all combinations of actions. This deep search evaluates the expected
% free energy under outcomes following an action and the ensuing hidden
% state. The average over  hidden states and future actions is then
% accumulated to provide the free energy expected under a particular course
% of action in the immediate future. Notice that the free energy expected
% under the next action is not the expected free energy following that
% action. In other words, there is a subtle distinction between taking an
% expectation under the posterior predictive distribution over outcomes and
% the expectation of free energy over outcomes. This scheme is
% sophisticated in the sense that the consequences of action for posterior
% beliefs enter into the evaluation of expected free energy.
%
% see also: DEM_demo_MDP_habits.m and spm_MPD_VB_X.m
%__________________________________________________________________________
 
% Karl Friston
% Copyright (C) 2008-2022 Wellcome Centre for Human Neuroimaging
 
% set up and preliminaries
%==========================================================================
rng('default')
  


% set of parameters (for comparison with actynf !)
%==========================================================================
% PROCESS
true_process_pHA = 1.0 ;
true_process_pinit = 1.0 ; % Useless here because this script fixes the outcomes
true_process_pwin = 0.98 ; % For a bit of noise !

% MODEL
true_model_pHA = 1.0 ;
true_model_pinit = 0.5 ;
true_model_pwin = 0.98 ;
true_model_context_belief = 0.5 ;
true_model_hint_conf = 2.0 ;
true_model_la = -4.0 ;
true_model_rs = 2.0 ;
true_alpha = 16.0 ;



% outcome probabilities: A
%--------------------------------------------------------------------------
% We start by specifying the likelihood mapping from hidden states
% to outcomes; where outcome can be exteroceptive or interoceptive: The
% exteroceptive outcomes A{1} provide cues about location and context,
% while interoceptive outcome A{2) denotes different levels of reward
%--------------------------------------------------------------------------
k = true_process_pHA ;
A{1}(:,:,1) = [...
    1 0 0 0;    % cue start
    0 1 0 0;    % cue left
    0 0 1 0;    % cue right
    0 0 0 k     % cue CS right
    0 0 0 1.0-k];   % cue CS left
A{1}(:,:,2) = [...
    1 0 0 0;    % cue start
    0 1 0 0;    % cue left
    0 0 1 0;    % cue right
    0 0 0 1.0-k     % cue CS right
    0 0 0 k];   % cue CS left
 
ka      = true_process_pwin;
kb      = 1 - ka;
A{2}(:,:,1) = [...
    1 0 0 1;    % reward neutral
    0 ka kb 0;    % reward positive
    0 kb ka 0];   % reward negative
A{2}(:,:,2) = [...
    1 0 0 1;    % reward neutral
    0 kb ka 0;    % reward positive
    0 ka kb 0];   % reward negative


% SUBJECT PERCEPTION MODEL !
v = 200;
x1 = true_model_hint_conf*(true_model_pHA);
x2 = true_model_hint_conf*(1-true_model_pHA);
a{1}(:,:,1) = [...
    v 0 0 0 ;    % cue start
    0 v 0 0 ;    % cue left
    0 0 v 0 ;    % cue right
    0 0 0 x1;    % cue CS right
    0 0 0 x2];   % cue CS left
a{1}(:,:,2) = [...
    v 0 0 0 ;    % cue start
    0 v 0 0 ;    % cue left
    0 0 v 0 ;    % cue right
    0 0 0 x2;    % cue CS right
    0 0 0 x1];   % cue CS left

ka      = 200*true_model_pwin;
kb      = 200*(1 - true_model_pwin);
a{2}(:,:,1) = [...
    v 0  0  v;    % reward neutral
    0 ka kb 0;    % reward positive
    0 kb ka 0];   % reward negative
a{2}(:,:,2) = [...
    v 0  0  v;    % reward neutral
    0 kb ka 0;    % reward positive
    0 ka kb 0];   % reward negative


% labeloutcome modalities
%--------------------------------------------------------------------------
label.modality{1} = 'where'; label.outcome{1} = {'start','left','right','cue R','cue L'};
label.modality{2} = 'what';  label.outcome{2} = {'=','+','-'};

% label hidden states
%--------------------------------------------------------------------------
label.factor{1}  = 'where';   label.name{1}   = {'start','left','right','cue'};
label.factor{2}  = 'context'; label.name{2}   = {'left','right'};
label.action{1}  = label.name{1};
 
% controlled transitions: B{u}
%--------------------------------------------------------------------------
% Next, we have to specify the probabilistic transitions of hidden states
% for each factor. Here, there are four actions taking the agent directly
% to each of the four locations (without absorbing right and left states).
%--------------------------------------------------------------------------
B{1}(:,:,1)  = [1 0 0 1; 0 1 0 0;0 0 1 0;0 0 0 0];
B{1}(:,:,2)  = [0 0 0 0; 1 1 0 1;0 0 1 0;0 0 0 0];
B{1}(:,:,3)  = [0 0 0 0; 0 1 0 0;1 0 1 1;0 0 0 0];
B{1}(:,:,4)  = [0 0 0 0; 0 1 0 0;0 0 1 0;1 0 0 1];
 
% context, which cannot be changed by action
%--------------------------------------------------------------------------
B{2}  = eye(2);
 
% priors: (utility) C
%--------------------------------------------------------------------------
% Finally, we have to specify the prior preferences in terms of log
% probabilities over outcomes. Here, the agent prefers rewards to losses
%--------------------------------------------------------------------------
T     = 3;
C{1}  = kron(ones(1,T),[0 0 0 0 0]');
C{2}  = kron(ones(1,T),[0  true_model_rs true_model_la]');
 
% now specify prior beliefs about initial states, in terms of counts. Here
% the hidden states are factorised into location and context:
%--------------------------------------------------------------------------
D{1} = [1 0 0 0]'; 
D{2} = [true_process_pinit (1.0-true_model_pinit)]';


d{1} = [128 1 1 1]';
k_conf = 4.0;
d{2} = [k_conf*true_model_context_belief k_conf*(1.0-true_model_context_belief)]';
 
 
% allowable actions (with an action for each hidden factor)
%--------------------------------------------------------------------------
U(:,1) = [1 2 3 4]';             % move to location
U(:,2) = 1;                      % stay in current context
 
% MDP Structure - this will be used to generate arrays for multiple trials
%==========================================================================
mdp.T = T;                       % two moves
mdp.U = U;                       % actions
mdp.a = a;                       % priors likelihood probabilities
mdp.A = A;                       % likelihood probabilities
mdp.B = B;                       % transition probabilities
mdp.C = C;                       % prior preferences
mdp.d = d;                       % prior over initial states
mdp.N = 2;                       % policy depth
% mdp.N = 1;
mdp.s = [1 1]';                  % true initial state

mdp.label = label;

% true initial states - with a couple of switches at the beginning
%--------------------------------------------------------------------------
i              = [1];          % change context in a couple of trials
[MDP(1:1)]    = deal(mdp);      % create structure array
[MDP(i).s]     = deal([1 2]');   % deal context changes
 
% Solve - an example with 32 trials to illustrate the transition from
% exploration to exploitation -  as beliefs about the context (i.e.,
% initial state) are accumulated
%==========================================================================

MDP  = spm_MDP_VB_XX_mod(MDP);

return
