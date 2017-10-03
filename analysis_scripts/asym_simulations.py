#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulates anti-symmetric (asym) measurements of two qubits.

Numerical tests of two qubits that simulates 2D histograms based on Poisson
distributions.

Author = Charles Baldwin


"""
import time
import sys

import numpy as np

sys.path.append('/installdir1/state_meas_tomo/')
sys.path.append('/installdir1/state_meas_tomo/analysis_scripts')
sys.path.append('/installdir1/state_meas_tomo/partial_tomography')
sys.path.append('/installdir2/lib/python3.4/site-packages/')
sys.path.append('/installdir2/lib/python3.4/site-packges/matlab')

import analysis
import analysis_tools


################## RELEVANT PARAMETERS AND TARGET STATES ######################
t = time.time()
N = 2               # number of qubits
bootstrap_iters = 2
bin_num = [2,2]

# Only asymmetric simulation code for 2 qubits exists
if N == 2:
    # target state prep
    state_vector = (1/np.sqrt(2))*np.array([1,0,0,1])
    
    # example observables
    obs1 = np.outer(state_vector, state_vector)
    obs2 = np.array([[0,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,0]])
    

target_rho = np.outer(state_vector, state_vector)
observables = np.dstack((obs1, obs2))


############################# DO ANALYSIS #####################################
SE = analysis.PartialMaxLike(seed=0)
SE.targets=[target_rho]  
SE.simulate(stateSim='asym', histSim='asym',fid = 0.99)
SE.autobinMaxLike(bin_num)

print('Running Tomgraphy...')
SE.tomography()

print('Running Bootstrap...')
bootstrap_fids = SE.bootstrap(iters=bootstrap_iters)
SE.bootstrapAnalysis()

## Uncomment to bound expectation values when run through terminal
#print('Finding expectation value bounds...')
#analysis_tools.expectationValues(SE, observables)

print('Elapsed Time: ' + str(time.time()- t) + ' seconds')