#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Outputs and plotting functions used in arXiv:17xx.xxxx

This module contains various functions used to create the figures and tables 
used in the paper as well as to analyse the expecation values.

Author = Charles Baldwin


"""
import pickle

import numpy as np
import scipy as sp
import scipy.stats as sts
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import gca
         
from histogram import Hist
import simulate_tools as st


def expectationValues(SE, observables):
    """
    Runs inferExpectation from analysis and saves to text file.
    
    For each observable calculates the bootstrap mean, semidefintie program 
    range, confidence intervals for both basic and bias-correcterd methods.
    
    Args: 
        observables: 3d array of observables
    
    """
    file = open('results-'+SE.name+'.txt','w') 
    
    # Heading material for each observable
    file.write('%3s %10s %10s %23s %23s %23s %23s \n' % 
               ('O', 'truth', 'estimate', 'SDP range', 'B.S. mean',  
               'Basic C.I.','BC C.I.'))
    file.write('--------------------------------'
               '-------------------------------')
    file.write('---------------------------------'
               '------------------------------')
    file.write('----------------------------------------------\n')

    try:
        num_obs = observables.shape[2]
    except:
        num_obs = 1

    SE.startMatlabEng()
    
    bound_list = {}
    flag_list = {}
    original_est = []
    true_val = []
    est_val = []
    bootstrap_mean = []
    basic_CI = []
    BC_CI = []
    
    for i in range(num_obs):
        print('Running observable %1i...' % (i+1))
        try:
            obs = observables[:,:,i]
        except:
            obs = observables

        bound_list[i], flag_list[i] = SE.inferExpectation(obs)
        
        # Calculates some important values
        original_est.append([bound_list[i][0,0], bound_list[i][0,1]])
        bootstrap_mean.append([np.mean(bound_list[i][1:,0]), 
                             np.mean(bound_list[i][1:,1])])
        true_val.append(np.real(np.trace(np.dot(st.makeRho(4,0.99),obs))))
        est_val.append(np.real(np.trace(np.dot(SE.est_rho[-1],obs))))
                        
        # Basic method confidence intervals
        basic_CI.append([2*original_est[-1][0]  
                            - np.percentile(bound_list[i][1:,0], 97.5),
                            2*original_est[-1][1]
                            - np.percentile(bound_list[i][1:,1], 2.5)])
        
        # BC method confidence intervals
        z0Lower = sts.norm.ppf(sts.percentileofscore(bound_list[i][1:,0], 
                                                     original_est[-1][0])/100)
        z0Upper = sts.norm.ppf(sts.percentileofscore(bound_list[i][1:,1],
                                                     original_est[-1][1])/100)
        BC_CI.append([np.real(np.percentile(bound_list[i][1:,0],
                         100*sts.norm.cdf(2*z0Lower + sts.norm.ppf(0.025)))),
                         np.real(np.percentile(bound_list[i][1:,1],
                         100*sts.norm.cdf(2*z0Upper + sts.norm.ppf(0.975))))])
        
        # outputs to shell
        print('original data bounds: (%f, %f)' %  
                  (original_est[-1][0], original_est[-1][1]))
        print('bootstrap data mean bounds: (%f,%f)' % 
                  (bootstrap_mean[-1][0], bootstrap_mean[-1][1]))
        print('Basic bootstrap confidence intertval: (%f,%f)' % 
                  (basic_CI[-1][0], basic_CI[-1][1]))
        
        # saves to .txt file
        file.write('%3i %10.4g %10.4g (%10.4g,%10.4g) (%10.4g,%10.4g) '
                   '(%10.4g,%10.4g) (%10.4g,%10.4g) \n' % 
                   (i+1, true_val[-1], true_val[-1], 
                    original_est[-1][0], original_est[-1][1], 
                    bootstrap_mean[-1][0], bootstrap_mean[-1][1],
                    basic_CI[-1][0], basic_CI[-1][1],                
                    BC_CI[-1][0], BC_CI[-1][1]))                 
                   
    file.close()
    SE.killMatlabEng()
    
    f2 = open('results-'+SE.name+'.out', 'wb')
    pickle.dump(bound_list,f2)
    f2.close()

        
def histogramPlots(SE):
    """Plots histograms for all reference and probe experiments."""
    
    fontsi = 12
    plt.figure(num=None, figsize=(6,8), dpi=80, facecolor='w', edgecolor='k')
    
    # the following is hard-coded to have 4 reference and 4 probing experiments
    
    # Reference experiments
    for ii in range(0,4):   
        ax1 = plt.subplot2grid((4,2), (ii, 0))
        a = SE.hists[ii].binbounds[0].tolist()
        plt.bar(SE.hists[ii].binbounds[0][0:-1], 
                SE.hists[ii].hist/SE.hists[ii].trials,
                width = [x - a[i - 1] for i, x in enumerate(a)][1:], 
                edgecolor='k',align = 'edge',color=(29/255,154/255,120/255))
    
        a = gca()
        mpl.rc('font', family = 'Times New Roman')
        #a.set_xticklabels([0,10,20,30,40,50,60], fontsize=fontsi-2)
        plt.axis([0, 60, 0, np.max(SE.hists[ii].hist/SE.hists[ii].trials)*1.05])
        plt.ylabel('Freq', fontsize=fontsi)
        
    plt.xlabel('Counts', fontsize=fontsi)
        
    # Probing histograms
    for ii in range(4,8):
        ax1 = plt.subplot2grid((4,2), (ii-4, 1))
        a = SE.hists[0].binbounds[0].tolist()
        plt.bar(SE.hists[ii].binbounds[0][0:-1], 
                SE.hists[ii].hist/SE.hists[ii].trials,
                width = [x - a[i - 1] for i, x in enumerate(a)][1:], 
                edgecolor='k', align = 'edge',color=(29/255,154/255,120/255))
    
        a = gca()
        mpl.rc('font', family = 'Times New Roman')
        #a.set_xticklabels([0,10,20,30,40,50,60], fontsize=fontsi-2)
        plt.axis([0, 60, 0, np.max(SE.hists[ii].hist/SE.hists[ii].trials)*1.05])

    plt.xlabel('Counts', fontsize=fontsi)
    
    plt.show()
    


def responsePlots(numTrials=5000, mu=[2,20,40]):
    """Plots histograms of states contained in POVM elements.""" 
    
    fontsi = 12
    
    dark_pops = [1,0,0] # only 2 dark state
    bright_pops = [0,0,1] # only 2 bright state
    mid_pops = [0,1,0]  # only 1 bright state
    
    # 2 dark histogram
    dark_hist = Hist()
    dark_counts = dark_hist.simPoisson(numTrials, pops=dark_pops, mu=mu) 
    plt.bar(dark_hist.binbounds[0][0:-1], dark_hist.hist/dark_hist.trials,
            width=1, color=(29/255,111/255,169/255),
            edgecolor='black', align='edge') 
    
    # 1 bright 1 dark histogram
    mid_hist = Hist()
    mid_counts = mid_hist.simPoisson(numTrials, pops=mid_pops, mu=mu) 
    plt.bar(mid_hist.binbounds[0][0:-1], mid_hist.hist/mid_hist.trials, 
            width=1, color=(241/255,157/255,25/255),
            edgecolor='black', align='edge') 
    
    # 2 bright histogram
    bright_hist = Hist()
    bright_counts = bright_hist.simPoisson(numTrials, pops=bright_pops, mu=mu) 
    plt.bar(bright_hist.binbounds[0][0:-1], bright_hist.hist/bright_hist.trials, 
            width=1, color=(183/255,73/255,25/255), 
            edgecolor='black', align='edge') 
    
    # plot everything
    a = gca()
    mpl.rc('font', family='Times New Roman')
    plt.axis([0, 60, 0, np.max(dark_hist.hist/dark_hist.trials)*1.05])
    plt.ylabel('Relative frequency',fontsize=fontsi)
    plt.xlabel('Counts',fontsize=fontsi)
    plt.legend(['0 bright', '1 bright', '2 bright'],loc = 5, fontsize=fontsi)
 
    
def figure2(SE):
    """Makes figure 2 from paper."""
    
    fontsi = 12
    plt.figure(num=None, figsize=(9,3.7), dpi=80, facecolor='w', edgecolor='k')
    
    axa = plt.subplot2grid((2,2), (0, 0), rowspan=2)
    axa.text(55, 0.27, r'(a)', fontsize=fontsi+2, fontweight='bold')
    responsePlots()
    
    # second reference histogram
    axb = plt.subplot2grid((2,2), (0, 1))
    axb.text(55, 0.065, r'(b)', fontsize=fontsi+2, fontweight='bold')
    a = SE.hists[1].binbounds[0].tolist()
    plt.bar(SE.hists[1].binbounds[0][0:-1], 
            SE.hists[1].hist/SE.hists[1].trials,
            width = [x - a[i-1] for i, x in enumerate(a)][1:], 
            edgecolor='k',align = 'edge',color=(29/255,154/255,120/255))
    
    a = gca()
    mpl.rc('font', family = 'Times New Roman')
    plt.axis([0, 60, 0, np.max(SE.hists[1].hist/SE.hists[1].trials)*1.05])
    
    # first probe histogram
    axc = plt.subplot2grid((2,2), (1, 1))
    axc.text(55, 0.12, r'(c)', fontsize=fontsi+2, fontweight='bold')
    a = SE.hists[4].binbounds[0].tolist()
    plt.bar(SE.hists[4].binbounds[0][0:-1], 
            SE.hists[4].hist/SE.hists[4].trials,
            width = [x - a[i-1] for i, x in enumerate(a)][1:], 
            edgecolor='k',align = 'edge',color=(29/255,154/255,120/255))
    
    a = gca()
    mpl.rc('font', family = 'Times New Roman')
    plt.axis([0, 60, 0, np.max(SE.hists[4].hist/SE.hists[4].trials)*1.05])
    plt.xlabel('Counts', fontsize=fontsi)

 
def figure5(SE,binnum):
    """Makes figure 5 from paper."""
    
    fontsi = 12
    plt.figure(num=None, figsize=(4.5,4), dpi=80, facecolor='w', edgecolor='k')
    
    # unbinned second reference histogram
    axa = plt.subplot2grid((2,1), (0,0))
    axa.text(55, 0.065, r'(a)', fontsize=fontsi+2, fontweight='bold')
    a = SE.hists[1].binbounds[0].tolist()
    plt.bar(SE.hists[1].binbounds[0][0:-1], 
            SE.hists[1].hist/SE.hists[1].trials,
            width = [x - a[i - 1] for i, x in enumerate(a)][1:], 
            edgecolor='k',align = 'edge',color=(29/255,154/255,120/255) )
    
    max_counts = max(SE.hists[1].binbounds[0][0:-1])
    
    a = gca()
    mpl.rc('font', family = 'Times New Roman')
    plt.axis([0,  60,
              0, np.max(SE.hists[1].hist/SE.hists[1].trials)*1.05])
    plt.xticks(fontsize = fontsi-2)
    plt.yticks(fontsize = fontsi-2)
    plt.ylabel('Relative frequency', fontsize=fontsi)
    
    # bin histogram
    SE.autobin(binnum)
    for i in range(binnum):
        plt.axvline(x=SE.hists[1].binbounds[0][i], 
                    linewidth=1, color = 'k')
    
    # binned second reference histogram
    axb = plt.subplot2grid((2,1), (1, 0))
    axb.text(55, 0.37, r'(b)', fontsize=fontsi+2, fontweight='bold')
    a = SE.hists[1].binbounds[0].tolist()
    plt.bar(SE.hists[1].binbounds[0][0:-1], 
            SE.hists[1].hist/SE.hists[1].trials,
            width = [x - a[i - 1] for i, x in enumerate(a)][1:], 
            edgecolor='k',align = 'edge',color=(29/255,154/255,120/255), )
    
    for i in range(binnum):
        plt.axvline(x=SE.hists[1].binbounds[0][i], 
                    linewidth=1, color = 'k')
        
    a = gca()
    mpl.rc('font', family = 'Times New Roman')
    plt.axis([0, 60, 
              0, np.max(SE.hists[1].hist/SE.hists[1].trials)*1.05])
    plt.ylabel('Relative frequency', fontsize=fontsi)
    plt.xlabel('Counts', fontsize=fontsi)
    
    
def likelihoodRatioPlot(bootstrap_ratios, meas_ratio):
    """Plots likelihood ratio test statistic.""" 
    
    plt.figure(num=None, figsize=(6,6), dpi=80, facecolor='w', edgecolor='k')
    fontsi = 14
    plt.hist(bootstrap_ratios)
    
    plt.ylabel('Counts', fontsize=fontsi)
    plt.xlabel('Likelihood ratio test statistic', fontsize=fontsi)
    plt.title('bootstrap loglikelihood ratio')
    
    plt.axvline(x=meas_ratio, linewidth=1, color = 'k')
    

def boundCorrelationPlot(SE, bootstrap_method = 'basic'):
    """Plots correlation between upper and lower bounds.""" 
    
    fontsi = 14
        
    bound_list = pickle.load(open('results-'+SE.name+'.out','rb'))
    num_obs = len(bound_list.keys())

    confidence_interval = []
    coverage = []
    for i in range(num_obs):
        # Basic method C.I.     
        if bootstrap_method == 'basic':
            confidence_interval.append([2*bound_list[i][0,0]  \
                                        - np.percentile(bound_list[i][1:,0], 97.5),
                                        2*bound_list[i][0,1]  \
                                        - np.percentile(bound_list[i][1:,1], 2.5)])
            coverage.append(np.sum(np.logical_and(bound_list[i][1:,1] <= confidence_interval[-1][1], 
                                        bound_list[i][1:,0] >= confidence_interval[-1][0]))/SE.boot_iters)
        
        # Percentile method C.I. 
        if bootstrap_method == 'percentile':
            confidence_interval.append([np.percentile(bound_list[i][1:,0],2.5),
                                  np.percentile(bound_list[i][1:,1],97.5)])
            coverage.append(np.sum(np.logical_and(bound_list[i][1:,1] <= confidence_interval[-1][1], 
                                        bound_list[i][1:,0] >= confidence_interval[-1][0]))/SE.boot_iters)
            
        # BC method C.I.
        if bootstrap_method == 'BC':
            z0_lower = sts.norm.ppf(sts.percentileofscore(bound_list[i][1:,0], 
                                bound_list[i][0,0])/100)
            z0_upper = sts.norm.ppf(sts.percentileofscore(bound_list[i][1:,1],
                                bound_list[i][0,1])/100)
            confidence_interval.append([np.percentile(bound_list[i][1:,0],
                                    100*sts.norm.cdf(2*z0_lower + sts.norm.ppf(0.025))),
                            np.percentile(bound_list[i][1:,1],
                                    100*sts.norm.cdf(2*z0_upper + sts.norm.ppf(0.975)))])
            coverage.append(np.sum(np.logical_and(bound_list[i][1:,1] <= confidence_interval[-1][1], 
                            bound_list[i][1:,0] >= confidence_interval[-1][0]))/SE.boot_iters)

        # Scatter plot of upper vs lower
        plt.figure(num=None, figsize=(8,6), dpi=80, facecolor='w', 
                   edgecolor='k')
            
        # plot bootstrap data and orignal data
        plt.scatter(bound_list[i][1:,0], bound_list[i][1:,1])
        plt.scatter(bound_list[i][0,0], bound_list[i][0,1])

        ax = plt.gca()
        ax.set_xlim(min(bound_list[i][:,0])-0.001, 
                    max(bound_list[i][:,0])+0.001)
        ax.set_ylim(min(bound_list[i][:,1])-0.001, 
                    max(bound_list[i][:,1])+0.001)
        plt.axvspan(confidence_interval[-1][0], 
                    max(bound_list[i][:,0])+0.001,
                    facecolor='0.3', alpha=0.2)
        plt.axhspan(min(bound_list[i][:,1])-0.001,
                    confidence_interval[-1][1],
                    facecolor='0.3', alpha=0.2)
            
        plt.ylabel('upper bound', fontsize=fontsi)
        plt.xlabel('lower bound', fontsize=fontsi)
        plt.title('Observable %1i, Coverage %4.3f' % (i+1,coverage[-1]), 
                  fontsize=fontsi)
            
        # histogram of lower bounds
        plt.figure(num=None, figsize=(8,6), dpi=80, facecolor='w', 
                   edgecolor='k')
        ax = plt.gca()
        plt.hist(bound_list[i][:,0])
        plt.axvspan(confidence_interval[-1][0], 
                    max(bound_list[i][:,0])+0.001,
                    facecolor='0.3', alpha=0.2)
        ax.set_xlim(min(bound_list[i][:,0])-0.001, 
                    max(bound_list[i][:,0])+0.001)
        plt.axvline(x=bound_list[i][0,0],linewidth=1, color = 'k')
        plt.xlabel('bootstrap bound', fontsize=fontsi)
        plt.ylabel('counts', fontsize=fontsi)
        plt.title('Observable %1i lower bound' % (i+1), fontsize=fontsi)
            
        # histogram of upper bounds
        plt.figure(num=None, figsize=(8,6), dpi=80, facecolor='w', 
                   edgecolor='k')
        ax = plt.gca()
        plt.hist(bound_list[i][:,1])
        plt.axvspan(min(bound_list[i][:,1])-0.001,
                    confidence_interval[-1][1],
                    facecolor='0.3', alpha=0.2)
        ax.set_xlim(min(bound_list[i][:,1])-0.001, 
                    max(bound_list[i][:,1])+0.001)
        plt.axvline(x=bound_list[i][0,1],linewidth=1, color = 'k')
        plt.xlabel('bootstrap bound', fontsize=fontsi)
        plt.ylabel('counts', fontsize=fontsi)
        plt.title('Observable %1i upper bound' % (i+1), fontsize=fontsi)

    return confidence_interval

def poisson(k, lamb):
    """Returns Poissonian distribution."""
    return (lamb**k/sp.misc.factorial(k)) * np.exp(-lamb)
    