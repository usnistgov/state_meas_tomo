"""
2D histogram tutorial for Joint Quantum State and Measurement Tomography

This script is an example use for 2d histogram objects as well as loading input 
data.

Example: 
    python tutorial.py
    
Author = Adam Keith
   
 
"""
import time
import sys

sys.path.append('/installdir1/state_meas_tomo/')
sys.path.append('/installdir1/state_meas_tomo/analysis_scripts')
sys.path.append('/installdir1/state_meas_tomo/partial_tomography')
sys.path.append('/installdir2/lib/python3.4/site-packages/')
sys.path.append('/installdir2/lib/python3.4/site-packges/matlab')

import numpy as np

import analysis
import histogram as hg        # Import Hist() class and helper functions
import simulate_tools as st   # Import tools for 2 ion example
import analysis_tools


############################## RELEVANT PARAMETERS ############################
SE_name = 'tutorial_save'  
N = 2   # number of qubits
bootstrap_iters = 20        # number of bootstrap repitions
bin_num = [2,2] # number of bins for each axis
seed_val = 0
t = time.time()
dim = 2**N


#################### STATES, OBSERVABLES, AND PROJECTORS ######################
# target state prep
state_vector = (1/np.sqrt(2))*np.array([1,0,0,1])
target_rho = np.outer(state_vector, state_vector)
    
# example observables
obs1 = np.outer(state_vector, state_vector)
obs2 = np.array([[0,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,0]])
# obs3 = ...
observables = np.dstack((obs1, obs2))  # add additional observables here

# Projectors are 1D elements to each subspace
num_sub = dim
state_map = np.eye(dim)
P_j = np.zeros((num_sub, dim, dim)) # Projectors defining subspaces
for j in range(num_sub):
    P_j[j, :, :] = np.diag(state_map[:, j])

           
######################### BUILD HISTOGRAMS FROM DATA ##########################
# Ion trappers, make a list of counts using countsFromExperimentClass()
# in general a list of counts or an existing histogram (array) will do
hists = []         # list of ALL histograms
input_state = []   # list of input states (labels for unknown states)
unitaries = []     # analysis unitaries

## Reference data (doesn't need any known unitary since individual addressing
## allows preperation of states with overlap of each underlying POVM element)
# Dark, Dark Reference
dark = np.zeros((dim, dim))+0j
dark[0, 0] = 1.0
c = np.loadtxt('hist0',delimiter=',')
h = hg.Hist(c)
hists.append(h)
input_state.append(dark)
unitaries.append(np.eye(dim))

# Dark, Bright Reference
db =  np.zeros((dim, dim))+0j
db[1,1] = 1.0
c = np.loadtxt('hist1',delimiter=',')
h = hg.Hist(c)
hists.append(h)
input_state.append(db)
unitaries.append(np.eye(dim))

# Bright, Dark Reference
bd =  np.zeros((dim, dim))+0j
bd[2,2] = 1.0
c = np.loadtxt('hist2',delimiter=',')
h = hg.Hist(c)
hists.append(h)
input_state.append(bd)
unitaries.append(np.eye(dim))

# Bright, Bright Reference
bright = np.zeros((dim, dim))+0j
bright[-1, -1] = 1.0
c = np.loadtxt('hist3',delimiter=',')
h = hg.Hist(c)
hists.append(h)
input_state.append(bright)
unitaries.append(np.eye(dim))

# Population Data
c = np.loadtxt('hist4',delimiter=',')
h = hg.Hist(c)
hists.append(h)
input_state.append(1)   # label "1" for first unknown density matrix
unitaries.append(np.eye(dim))

## Probing data
phases = [0.0, 0.7853981633974483, 1.5707963267948966, 2.356194490192345, 
          3.141592653589793, 3.9269908169872414, 4.71238898038469, 
          5.497787143782138]
for i in range(5, 13):
    c = np.loadtxt('hist' + str(i),delimiter=',')
    h = hg.Hist(c)
    hists.append(h)
    input_state.append(1)   # label "1" for first unknown density matrix
    unitaries.append(st.pulseSequence([np.pi/2], phi=[phases[i-5]], N=N))
    

############################# DO ANALYSIS #####################################
SE = analysis.PartialMaxLike(hists, input_state, unitaries, P_j, train_frac=0.1,
                             name=SE_name, targets=[target_rho], seed=seed_val)
SE.autobinMaxLike(bin_num) 


print('Running Tomography...')
inferred_fidelities = SE.tomography()
## Note SE.estRho is a list of inferred density matrices (not just one!)

print('Running Bootstrap...')
bootstrap_fids = SE.bootstrap(iters=bootstrap_iters)
SE.bootstrapAnalysis()

## Uncomment to bound expectation values when run through terminal
#print('Finding expectation value bounds...')
#analysis_tools.expectationValues(SE, observables)

print('Elapsed Time: ' + str(time.time()- t) + ' seconds')
