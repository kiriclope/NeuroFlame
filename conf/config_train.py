###################################################
# Parameters 
###################################################
# output different prompts for debugging purpose
VERBOSE: 0
# device to be used cuda or cpu
DEVICE: cuda
# float precision
FLOAT_PRECISION: 32
# 1 to save the data to DATA_PATH
SAVE_DATA: 1

N_BATCH: 1
# Time step in s
DT: 0.001
# total simulation time in s
DURATION: 5.5
# time to start saving simulation files
T_STEADY: 0.5
# Saving to files every T_WINDOW
T_WINDOW: 0.25

# record only endpoint
REC_LAST_ONLY: 1

##########################################
# Parameters for the stimulus presentation
##########################################
# stimulus as a cosine shape
TASK: 'dual'
# time for stimuli onset in s 
T_STIM_ON: [1.0, 4.0]
# time for stimuli offset in s 
T_STIM_OFF: [2.0, 5.0]
# stimuli strengths
I0: [.1, .1]
# stimuli phases
PHI0: [180.0, 90.0]
# stimuli tuning
SIGMA0: [1.0, 1.0]

##########################################
# Network parameters
##########################################
# number of populations
N_POP: 2
# number of neurons
N_NEURON: 10000
# number of average presynaptic input (set to 1 if all to all)
K: 500.0
# fraction of neuron in each population (must sum to 1)
frac: [0.8, 0.2]

# Network Dynamic
##########################################
# Transfert Function
# set to 0 for threshold linear, 1 for sigmoid
TF_TYPE: 'relu'

# Dynamics of the rates
# set to 0 if instantaneous rates, 1 if exponentially decaying
RATE_DYN: 1
# rate time constants
TAU: [0.020, 0.010]

# Dynamics of the recurrent inputs
# set to 0 if instantaneous, 1 if exponentially decaying
SYN_DYN: 1
# Synaptic time constants for each population
TAU_SYN: [.004, .002]

# Feedforward inputs dynamics
# Variance of the noise
VAR_FF: [30, 30]

# threshold
THRESH: [1.0, 1.0]

# Network's gain
GAIN: 1.0

# Synaptic strengths           
Jab: [34., -1.5, 1, -1]
# External inputs strengths
Ja0: [2.0, 1.0]
# External rate
M0: 10.0

# To add an attentional switch
# if BUMP_SWITCH[i] == 1 it sets Iext[i] to zero before stimulus presentation
BUMP_SWITCH: [1, 0]

####################
# Plasticity
####################
# adds short term plasticity
IF_STP: 1
USE: 0.03
TAU_FAC: 1.0
TAU_REC: 0.25

##############
# Connectivity
##############
# seed for connectivity None or float
SEED: 0
# CON_TYPE is all2all or sparse
# By default the matrix is a random sparse matrix
# 'all2all' gives an all to all matrix
# PROBA_TYPE can be 'cosine', 'cosine_spec' or 'lr'
# By default the matrix is a random sparse matrix
# 'cos' gives a sparse matrix with strong cosine structure
# 'spec_cos' gives a sparse matrix with weak cosine structure
# 'all2all' +'cos' gives an all to all with cosine shape

# sets probabilities of connections' shape
CON_TYPE: 'sparse'
PROBA_TYPE: ['lr', 'None', 'None', 'None']

# strength of the asymmetries if all to all
SIGMA: [0.0, 0.0, 0.0, 0.0]
# tuning of the recurrent connections
KAPPA: [4.0, 0.0, 0.0, 0.0]
# phase of the connectivity
PHASE: 0.0

##########
# Low rank
##########
RANK: 2

LR_EVAL_WIN: 1
LR_TRAIN: 1
LR_MEAN: [0.0, 0.0]
LR_COV: [[1.0, 0.0],[0.0, 1.0]]
