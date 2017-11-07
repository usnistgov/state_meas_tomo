# -*- coding: utf-8 -*-
"""
Script for importing CNOT data from .csv files into python and running Adam's
joint state-measurement tomography code

@author: Karl Mayer
"""

import time
import sys

sys.path.append('C:/Users/khm/.spyder2-py3/state_meas_tomo/')
sys.path.append('C:/Users/khm/.spyder2-py3/state_meas_tomo/analysis_scripts')
sys.path.append('C:/Users/khm/.spyder2-py3/state_meas_tomo/partial_tomography')
sys.path.append('C:/Users/khm/.spyder2-py3/lib/site-packges')
sys.path.append('C:/Users/khm/.spyder2-py3/lib/site-packges/matlab')

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
# target state preps
DZ = np.array([0,1])
UX = np.array([1,1])*(1/np.sqrt(2))
UY = np.array([1,1j])*(1/np.sqrt(2))
UZ = np.array([1,0])
input_states = [DZ, UX, UY, UZ]
CNOT = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
target_rho = []
for s1 in range(4):
    for s2 in range(4):
        new_target = np.dot(CNOT,np.kron(input_states[s1],input_states[s2]))
        new_target_rho = np.outer(new_target, np.conj(new_target))
        target_rho.append(new_target_rho)
        
    
# example observables
#obs1 = np.outer(state_vector, state_vector)
#obs2 = np.array([[0,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,0]])
# obs3 = ...
#observables = np.dstack((obs1, obs2))  # add additional observables here

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

brightFile = 'Bright_Reference_2017-10-23--18.28.28.014.csv'
darkFile = 'Dark_Reference_2017-10-23--18.29.40.574.csv'
c_bright = np.loadtxt(brightFile,delimiter=',')
c_bright = np.delete(c_bright, [0,1], 1)
c_dark = np.loadtxt(darkFile,delimiter=',')
c_dark = np.delete(c_dark, [0,1], 1)
L_min = min([len(c_dark), len(c_bright)]) # in case bright and dark histograms have
                                          # different number of elements
# Bright, Bright Reference
bb = np.zeros((dim, dim))+0j
bb[0, 0] = 1.0
c = c_bright
h = hg.Hist(c)
hists.append(h)
input_state.append(bb)
unitaries.append(np.eye(dim))

# Bright, Dark Reference
bd =  np.zeros((dim, dim))+0j
bd[1,1] = 1.0
c = np.transpose([c_bright[0:L_min,0], c_dark[0:L_min,1]])
h = hg.Hist(c)
hists.append(h)
input_state.append(bd)
unitaries.append(np.eye(dim))

# Dark, Bright Reference
db =  np.zeros((dim, dim))+0j
db[2,2] = 1.0
c = np.transpose([c_dark[0:L_min,0], c_bright[0:L_min,1]])
h = hg.Hist(c)
hists.append(h)
input_state.append(db)
unitaries.append(np.eye(dim))

# Dark, Dark Reference
dd = np.zeros((dim, dim))+0j
dd[3, 3] = 1.0
c = c_dark
h = hg.Hist(c)
hists.append(h)
input_state.append(dd)
unitaries.append(np.eye(dim))

# Probing Data
import glob
pi = np.pi

temp_state_label = 'temp label' # to be used in for loop below
num_label = 0

for file in glob.glob('*.csv'):
    # exclude 'Bright_Ref', 'Dark_Ref', and 'Pumping_Ref' files 
    if file[2] == '_':
        c = np.loadtxt(file, delimiter=',')
        c = np.delete(c, [0,1], 1) # delete first two columns
        h = hg.Hist(c)
        hists.append(h)
        state_label = file[0:5]
        if state_label != temp_state_label:
            temp_state_label, num_label = state_label, num_label+1
            
        input_state.append(num_label) # add label for unknown density matrix
        # define unitary to append to unitaries        
        meas_setting = file[6:9]
        if meas_setting[0] is 'X':
            theta1, phi1 = pi/2, 3*pi/2
            
        if meas_setting[0] is 'Y':
            theta1, phi1 = pi/2, 0
        
        if meas_setting[0] is 'Z':
            theta1, phi1 = 0, 0
        
        if meas_setting[2] is 'X':
            theta2, phi2 = pi/2, 3*pi/2
            
        if meas_setting[2] is 'Y':
            theta2, phi2 = pi/2, 0
        
        if meas_setting[2] is 'Z':
            theta2, phi2 = 0, 0
            
        unitaries.append(st.pulseSequence([theta1, theta2], [phi1, phi2], N=N))
        
############################# DO ANALYSIS #####################################
SE = analysis.PartialMaxLike(hists, input_state, unitaries, P_j, train_frac=0.1,
                             name=SE_name, targets=target_rho, seed=seed_val)
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
