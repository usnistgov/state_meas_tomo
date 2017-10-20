#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulates symmetric (sym) measurements of two qubits.

This script generates the numerical tests used in the paper and corresponding
figures.

Example:
    python paper_simulations.py
    

Author = Charles Baldwin


"""
import sys
import time

import numpy as np

sys.path.append('/installdir1/state_meas_tomo/')
sys.path.append('/installdir1/state_meas_tomo/analysis_scripts')
sys.path.append('/installdir1/state_meas_tomo/partial_tomography')
sys.path.append('/installdir2/lib/python3.4/site-packages/')
sys.path.append('/installdir2/lib/python3.4/site-packges/matlab')

import analysis_tools
import analysis
   

################## RELEVANT PARAMETERS AND TARGET STATES ######################
T = time.time()
N = 2 # number of qubits
bootstrap_iter = 10
bin_num = 5
p_mix = 0.99  # mixture with target state and max-mixed state
seed_val = 0

if N == 1:
    # target state prep
    state_vector = (1/np.sqrt(2))*np.array([1,1])
    
    # example observables
    obs1 = np.outer(state_vector, state_vector)
    obs2 = (1/2)*np.array([[1,np.e**(1j*np.pi/4)],[np.e**(-1j*np.pi/4),1]])
    # obs3 = ...
    
if N == 2:
    # target state prep
    state_vector = (1/np.sqrt(2))*np.array([1,0,0,1])
    
    # example observables
    obs1 = np.outer(state_vector, state_vector)
    obs2 = np.array([[0,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,0]])
    # obs3 = ...

target_rho = np.outer(state_vector, state_vector)
observables = np.dstack((obs1, obs2))   # add additional observables here

# set up analysis
SE = analysis.PartialMaxLike(seed=seed_val)
SE.targets=[target_rho]  
SE.simulate(fid=p_mix)
           
########################### PLOT HISTOGRAMS ###################################

# plot unbinned histograms
analysis_tools.figure2(SE)    # hard-coded for N = 2
analysis_tools.figure5(SE, bin_num)

############################# DO ANALYSIS #####################################

# bin histograms
SE.autobinMaxLike(bin_num)

print('Running Tomgraphy...')
SE.tomography()

print('Running Bootstrap...')
bootstrap_fids = SE.bootstrap(method='parametric', iters=bootstrap_iter)
SE.bootstrapAnalysis()

## Uncomment to bound expectation values when run through terminal
#print('Finding expectation value bounds...')
#analysis_tools.expectationValues(SE, observables, p_mix)

print('Elapsed Time: ' + str(time.time()- T) + ' seconds')