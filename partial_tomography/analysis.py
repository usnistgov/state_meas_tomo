"""
Analysis of partial maximum likelihood.

These classes contain the parameters used in the joint quantum state and 
measurement tomography procedure introduced in ...


Problems:
    Requires Matlab 2015 engine or later
        use build a conda package from Matlab's setup.py
        Matlab requires YALMIP and OptiToolbox
    Autosaves analysis object after tomography and/or bootstrap
    This code works for an arbitrary number of density matrices to estimate
        so remember that if you are only estimating one, its a list of one
        density matrix, i.e. self.est_rho[0] is a numpy array with
        shape (dim, dim)
        
Author = Adam Keith

"""


import copy
import pickle
import datetime
import os
# import matlab.engine  # uncomment for expectation-value bounding

import numpy as np
import scipy.linalg as linalg
from scipy.stats import percentileofscore

import histogram as hg        # Import Hist() class and helper functions
import tomography as tm       # Tomography code for PQSTMLE
import simulate_tools as st   # tools for simulating trapped ions


class HistList(object):
    """
    A collection of histogram.Hist() instances
    
    Mannages array of histogram.Hist() instances and confirms that several 
    parameters, like bin bounds (bin_bounds), dimension (dim_hists),
    are the same for all Hist().
    
    Attributes:
        hists:      array of Hist() instances
        num_hists:  number of Hist() instances
        dim_hist:   dimension of histogram
        mbins:      Array of number of bins in each dimension of histogram
    
    """

    def __init__(self, hists=None, bin_flag=True):
        """
        Construct a HistList instance from a list of histograms

        Args:
            hists:    list of histogram.Hist() objects
            bin_flag: flag to automatically bin all histograms
                      in integer spacing (don't want to do this for
                      parameteric resampling)
                      
        """
        # Hists given, do things
        if hists is not None:         
            self.hists = np.array(hists)  
            self.num_hists = len(hists)
            self.confirmEqSpecies()
            if bin_flag is True:
                # Force all histograms to same binning
                HistList.rebinList(self)
    
    def confirmEqSpecies(self):
        """
        Confirms all elements of hists have same dimension (dim_hist). 
        
        Raises:
            TypeError: hists objects do not have same number of species
        
        """
        temp = self.hists[0].dim_hist
        for r in self.hists:
            if r.dim_hist != temp:
                raise TypeError("Histograms are not all the "
                                "same number of species")
                return
        self.dim_hist = temp    # Number of distinguishable species

    def findCountRangeTotal(self):
        """Finds largest and smallest counts of all hists along each dim."""
        
        max_counts = []
        min_counts = []
        for r in self.hists:
            r.findCountRange()  # update maxcounts of each histogram
            max_counts.append(r.max_counts)
            min_counts.append(r.min_counts)
        self.max_counts = np.amax(max_counts, 0)
        self.min_counts = np.amin(min_counts, 0)

    def convertListOfHistsToMatrix(self):
        """
        Stacks hists vertically as numpy array
        
        In the paper we refer to histograms as the matrices that are created
        after the stacking
        
        Returns:
            hist_stack: Matrix of histograms (if no hists then empy array)
        
        """
        # empty list
        if not np.array(self.hists).size:  
            return np.array([])
        
        hist_stack = [x.hist1D() for x in self.hists]
        hist_stack = np.vstack(hist_stack)
        
        return hist_stack
    
    def rebinList(self, new_bin_bounds=None, bin_index=None):
        """
        Rebin list of histograms in the same way

        Args:
            new_bin_bounds: New bin boundaries to bin all elements of hists
                            See Hist().rebin()
                            Defualts to None, bin all refs in each dimension by
                            integers from each dimension's mincount to maxcount
            bin_index:      Indices of hists that you would like to bin
                            default is all of the histograms
        Returns:
            bin_bounds:     equal to new_bin_bounds
                            
        """
        # Bin histogram along each axis from min_count to max_count+2
        if new_bin_bounds is None:
            self.findCountRangeTotal()
            new_bin_bounds = []
            for i in range(self.dim_hist):
                new_bin_bounds.append(np.arange(self.min_counts[i],
                                              self.max_counts[i]+2))
        if bin_index is None:
            bin_index = range(self.num_hists)

        # Do the binning with Hist.rebin() method in histogram
        for i in bin_index:
            self.hists[i].rebin(new_bin_bounds)
        self.bin_bounds = self.hists[0].bin_bounds  # pass up the bin_bounds
        self.mbins = self.hists[0].bins           # number of bins in each dim
        self.bins = np.prod(self.mbins)           # number of bins flattened
        
        return self.bin_bounds


class AnalysisParameters(HistList):
    """
    List of histograms along with some parameters describing system.
    
    The AnalysisParameters object is a subclass of the HistList object that
    also includes useful parameter that describe the system. It is used for
    extracting the binning rule from the training data.
    
    Attributes:
        hists:          Array of Hist() objects.
        ref_ind:        Array of indeces that for the references experiments
        ref_num:        Number of reference expeirments
        probe_ind:      Array of indeces for the probing experiments
        probe_ind_data: Array of indeces for each probing experiment for each 
                        unknown state
        probe_num_each: Number of probing experiments per unknown state
        probe_num:      Total number of probing experiments
        input_state:    Array of input density matrices 
        state_label:    Array of labels for reference and probing experiments
        unitaries:      Array of known unitaries applied to input states.
        P_j:            Array of subspace projector matrices.
        num_sub:        Number of subspaces (number of P_j's)
        dim:            Dimension of Hilbert space
        est_Q:          Estimated value of transition matrix Q
        bin_flag:       Flag to automatically bin all reference histograms
                        in integer spacing.
        mbins:          Array of number of bins in each dimension of histogram
        name:           String describing these AnalaysisParameters instance
        
    
    """
    def __init__(self, hists=None, input_state=None, unitaries=None,
                 P_j=None, bin_flag=True, name=None):
        """
        Constructs an instance from HistList and populations
        
        AnalysisParameters is a subclass of HistList that also contains, the 
        input states, known unitaries, and projectors. The AnalysisParameters
        object creates other useful parameters to be used in the tomography
        procedure but does not run the tomography.

        Args:
            hists:         Array of Hist() objects.
            input_state:   Array of input density matrices (as numpy array)
                           before unitary was applied. If state is unknown,
                           input_state = i to label all histograms that
                           have the same initial state i>0 (0 is reserved for
                           references). 
            unitaries:     Array of known unitaries applied to input states.
            P_j:           Array of underlying POVM elements.
            bin_flag:      True then automatically bin all reference histograms
                           in integer spacing.
            name:          String describing these AnalaysisParameters instance
                     
        Raises:
            ValueError: 
                
        Assumptions:
            -all input_states and unitaries have the same dimension and
             are square
            -all histograms have the same dimension
            -all histograms with known input_state (value not equal to an
             integer) are references
            -hists index corresponds with input_state and unitaries index
            -References are used for arbitrary number of states that need to be
             estimated
             This does not preserve the order of the input 

        """
        super().__init__(hists, bin_flag)
        
        if hists is None:
            return

        self.name = name 
        # default is today's date        
        if self.name is None:    
            self.name = str(datetime.date.today())

        self.input_state = input_state            
        self.unitaries = np.array(unitaries)      
        self.P_j = np.array(P_j)                  
        self.num_subs = self.P_j.shape[0]    
        self.dim = self.unitaries.shape[-1]       
        self.est_Q = None                         
        self.measurementProjectors()             
        
        assert self.num_hists == len(input_state) == self.unitaries.shape[0]
        
        # Creates state_label array to label reference histogram as 0
        self.state_label = copy.deepcopy(self.input_state)
        for i in range(self.num_hists):
            if np.array(self.input_state[i]).size > 1:
                self.state_label[i] = 0
            elif self.state_label[i] == 0:
                raise ValueError("State Label 0 is reserved for reference "
                                 "histograms")
                
        self.input_state = np.array(self.input_state)
        self.state_label = np.array(self.state_label)
        self.distinguishHists()
        
        if bin_flag:
            self.rebinList()  # merging removes binning           
        
    def distinguishHists(self):
        """
        Distinguishes between reference and probing histograms.
        
        Raises:
            ValueError: There are no reference experiments
        
        """
        ## Reference experiments indices
        self.ref_ind = np.where(self.state_label == 0)[0]
        self.ref_num = self.ref_ind.size
        if self.ref_num == 0:
            raise ValueError('No reference histograms == no tomography :(')
        self.referencePopulations()

        ## Probing experiments indices
        # Relabel state_labels so that probe labels are in ascending order    
        labels = np.unique(self.state_label)
        temp = np.arange(labels.size)
        # now, subtracting 1 from state_label will give index on rhos IF probe
        self.state_label = np.array([temp[labels == self.state_label[i]][0] 
                                    for i in range(self.num_hists)])

        # Probe indices regardless of unknown density matrix
        self.probe_ind = np.nonzero(self.state_label)[0]
        probe_labels, num = np.unique(self.state_label, return_counts=True)
        self.probe_num_each = num[probe_labels != 0]  # number of hists for each 
        self.probe_num = np.sum(self.probe_num_each)  # total probe hists
        probe_labels = probe_labels[probe_labels != 0]  # probe labels
        self.num_rhos = probe_labels.size       # number of unknown states

        # For each unknown rho, index all probe histograms 
        probe_state_label = self.state_label[self.probe_ind]
        self.probe_ind_data = [np.where(probe_state_label == probe_labels[k])[0]
                               for k in range(self.num_rhos)]

    def autobinParams(self, nbins=None, strategy='subspace'):
        """
        Find binning rule based on mutual information heuristic.

        Use mutual information between reference states and observed counts
        with target state equal to the collection of all observed counts from
        references.

        For multi ion binning, bin each marginal distribution as a single ion
        then use binning along each dimension to form multidimensional
        histogram

        Known issues:
            Cannot bin for nbins=1 or more than number of raw_hist_bins

        Args:
            nbins:     Number of final bins (in each dimension)
            strategy:  Different strategies to bin. 
                       'subspace' estimates subspaces distributions,
                       and builds mixed state using them. 
                       'references' builds mixed state by adding all references
                       together
        Returns:
            mut_info_ratio: ratio of mutual information of new binning to
                            no binning along each axis.
                            
        """
        assert(strategy is 'subspace' or strategy is 'references')
        if self.ref_num == 0:
            return

        # Ensure for nbins=None we have at least as many bins as POVM elements
        if nbins is None:
            if self.dim_hist == 1:
                nbins = self.num_subs
            else:
                nbins = int(np.ceil(np.log2(self.num_subs)))

        if isinstance(nbins, int):
            nbins = nbins*np.ones(self.dim_hist, dtype=int)

        # return to default binning (uniform integer binning)
        self.rebinParams(bin_index=self.ref_ind)
        
        # If initialQ estimate failed, change strategy
        if self.est_Q is None and strategy is 'subspace':
            print('Since Q cannot be estimated, change binning strategy to '
                  'use reference histograms')
            strategy = 'references'
        
        pc, ps, p_cgs, states = self.binningStrategy(strategy)
        bin_bounds_list = []  
        mut_info_ratio = [] 
        pc = pc.reshape(self.mbins)  # same shape as underyling histograms
        for n in range(self.dim_hist):
            # Calculates marginal probability distributions 
            pc_n = hg.marginal(pc, n)  
            p_cgs_n = np.array([hg.marginal(p_cgs[i].reshape(self.mbins), n)
                                for i in range(states)])
            
            mut_info_ratio.append(mutualInfo(pc_n, ps, p_cgs_n))
            
            # for each dimension, bin marginals
            lb = self.min_counts[n]
            ub = self.max_counts[n]+1
            
            bin_div_choices = np.arange(lb+1, ub)
            bin_bounds = np.array([lb, ub])  
            for bin_count in range(2, nbins[n]+1):
                # Remove bin_bounds already used
                not_used = np.in1d(bin_div_choices, bin_bounds, invert=True)
                bin_div_choices = bin_div_choices[not_used]
                
                # Add a new bin and calculate new mutual information
                infos = []
                for bin_divider in bin_div_choices:
                    temp_bnds = np.append(bin_bounds, bin_divider)
                    temp_bnds = np.sort(temp_bnds)
                    # Bin distributions
                    count_prob_bin = binDists(pc_n, temp_bnds)
                    count_prob_givenR_bin = binDists(p_cgs_n, temp_bnds)
                    infos.append(mutualInfo(count_prob_bin, ps,
                                            count_prob_givenR_bin))

                # find maximum mutual info
                ind = np.argmax(infos)
                new_bin_div = bin_div_choices[ind]  
                bin_bounds = np.append(bin_bounds, new_bin_div)  
                bin_bounds = np.sort(bin_bounds)
            bin_bounds_list.append(bin_bounds)  
            mut_info_ratio[-1] = np.amax(infos)/mut_info_ratio[-1]

        # Applies binning rule to histograms and Q matrix
        self.rebinParams(bin_bounds_list)
        
        return mut_info_ratio
    
    def rebinParams(self, new_bin_bounds=None, bin_index=None):
        """Rebin all elements in hists and estimate of transition matrix"""
        
        HistList.rebinList(self, new_bin_bounds=new_bin_bounds, 
                           bin_index=bin_index)
        
        if self.bins < self.num_subs:
            print('Warning: Fewer bins than distinguishable subspaces')
            
        # arbitrary rebinning may not be possible with estimated distributions
        # so quick reestimate the transition matrix
        self.initialQ()
        
        return self.bin_bounds

    def binningStrategy(self, strategy='subspace'):
        """
        Creates probability distributions for binning 

        Args:
            strategy:   Different strategies to bin. 
                        'subspace' estimates distribution based on histograms
                        'references' estimates distrubtion based on
                        transition matrix
        
        Returns:
            ps:         Probability distribution of subspace 
            pc:         Probability distribution of counts
            p_cgr:      Probability distribution of counts given subspace
            states:     number of reference states

        """ 
        self.bin_strat = strategy
        
        if strategy is 'references':
            ps = [self.hists[k].trials for k in self.ref_ind]
            
            # Adds all reference histograms and divide by total trials
            pc = np.zeros(self.mbins)
            for k in self.ref_ind:
                pc += self.hists[k].hist
            pc = pc/np.sum(ps)
            ps = ps/np.sum(ps)
            
            # Calculates probability of counts given reference histogram
            p_cgs = np.array([self.hists[k].hist/self.hists[k].trials
                              for k in self.ref_ind])
            
            states = self.ref_num
            
        elif strategy is 'subspace':
            # probability of observing count, 1D histogram
            pc = self.makeExempHist()  
            
            ps = np.ones(self.num_subs)/self.num_subs
            p_cgs = self.est_Q   
            states = self.num_subs
            
        return (pc, ps, p_cgs, states)

    def mutualInfoRatio(self):
        """
        Calculates mutual information ratio between new binning and original.
        
        Assumptions:
            autobinParams() has already been called
            
        Returns:
            mut_info_ratio: mutual information ratio between original strategy
                            and binning strategy found in autobinParams()
        
        """
        auto_bin_bounds = self.bin_bounds  
        
        # rebins to no bins to calculate original mutual info
        self.rebinParams()  
        pc, ps, p_cgs, states = self.binningStrategy(self.bin_strat)
        mut_info_ratio = []
        pc = pc.reshape(self.mbins)
        for n in range(self.dim_hist):
            # marginal probability distributions
            pc_n = hg.marginal(pc, n)  
            p_cgs_n = np.array([hg.marginal(p_cgs[i].reshape(self.mbins), n)
                                for i in range(states)])
            
            # original mutual information saved as element of mut_info_ratio
            mut_info_ratio.append(mutualInfo(pc_n, ps, p_cgs_n))
            
            # Calculates mutual information of bined version
            count_prob_bin = binDists(pc_n, auto_bin_bounds[n])
            count_prob_givenR_bin = binDists(p_cgs_n, auto_bin_bounds[n])
            autobin_info = mutualInfo(count_prob_bin, ps,
                                      count_prob_givenR_bin)
            
            mut_info_ratio[-1] = autobin_info/mut_info_ratio[-1]
            
        # Changes back to autobinParams() auto_bin_bounds
        self.rebinParams(auto_bin_bounds)  
        
        return mut_info_ratio
    
    def measurementProjectors(self):
        """Underlying POVM elements for engineered measurements."""
        
        # Rotated projectors (measurement with settings)
        self.P_ij = np.zeros((self.num_hists, self.num_subs,
                              self.dim, self.dim))+0j
        for i in range(self.num_hists):
            for j in range(self.num_subs):
                self.P_ij[i,j,:,:] = (self.unitaries[i].conj().T).dot(
                                      self.P_j[j]).dot(self.unitaries[i]) 
                
    def initialQ(self):
        """Initial estimate of transition matrix."""
        
        H = self.convertListOfHistsToMatrix()[self.ref_ind]
        H = H/np.sum(H, 1)[:, np.newaxis]
        try:
            Q = np.linalg.inv(np.transpose(self.pops).dot(
                              self.pops)).dot(np.transpose(self.pops)).dot(H)
        except np.linalg.LinAlgError:  # probably indistinguishable pure states
            print('Warning: P is singular, cannot estimate Q by inversion.')
            return None
        #q[q < 0] = 1e-9  # "zero" out negative probabilities (make small)
        Q[Q < 0] = 0  # "zero" out negative probabilities (make small)
        Q += 1e-9  # "zero" out negative probabilities (make small)
        Q = Q/np.sum(Q, 1)[:, np.newaxis]
        self.est_Q = Q
        
        return self.est_Q

    def makeExempHist(self, p_exemp=None):
        """
        Make 1D histogram for examplar state used for constructing bin rule.

        Args:
            p_exemp:      Desired populations. If None, equal probability of each
                          underlying POVM outcome
        Returns:
            hist_exemp:   Generated histogram for examplar state.
        """
        
        if p_exemp is None:
            p_exemp = np.ones(self.num_subs)/self.num_subs

        if self.est_Q is None:
            self.initialQ()

        hist_exemp = np.zeros(self.est_Q.shape[1])
        for i in range(self.num_subs):
            hist_exemp += p_exemp[i]*self.est_Q[i]
            
        return hist_exemp
    
    def referencePopulations(self):
        """Calculates populations for reference histograms."""
        
        rho = np.array([self.input_state[self.ref_ind][k]
                        for k in range(self.ref_num)])
        self.pops = np.real(np.trace(np.einsum('iak,ijkb->ijab', rho,
                                               self.P_ij[self.ref_ind]),
                                               axis1=-1, axis2=-2))


class PartialMaxLike(AnalysisParameters):
    """
    An AnalysisParameters object with methods for tomography.
    
    Additional methods allow for the implementation of alternating maximum
    likelihood estimation, bootstrap resampling, and inferring expectation
    values. 
    
    Attributes:
        seed:                   seed for the random number generator
        train_frac:             fraction of reference histograms used for 
                                training
        bin_strat:              binning strategy for training data
        measure:                function that 'measures' estimated density
                                matrices
        targets:                list of target density matrices, same size as 
                                unknown density matrices
        tom:                    instance of tomography object used for data
        est_rho:                estimated unknown state's density matrices
        est_Q:                  estimated transition matrix
        out_value:              value of measure for estimated density matrix
        loglike_final_est:      final value of loglike for data
        boot_iters:             number of boostrap iterations
        bootstrap_log:          array of PartialMaxLike instances for each 
                                bootstrap iteration
        bootstrap_value:        value of measure for each bootstrap iter
        loglike_final_boot:     final value of loglike for each bootstrap iter
        likelihood_percentile:  percentile of data w.r.t. bootstraps
        conf_int:               confidence interval for measure
        autosave:               if True, will save itself after tomography 
                                and bootstrap
                                
                                
    """
    def __init__(self, hists=None, input_state=None, unitaries=None,
                 P_j=None, train_frac=0.1, bin_flag=True,
                 name=None, seed=None, measure=None, targets=None,
                 autosave=True):
        """
        Constructs an instance of PartialMaxLike.
        
        Args:
            hists:          array of histograms
            input_state:    array of fiducial states for ref experiments and
                            integers for probing experiments
            unitaries:      known unitaries
            P_j:            underlying POVM elements
            train_frac:     fraction of reference histogram to use for 
                            training data
            bin_flag:       if true automatically rebin all reference 
                            histograms
            name:           name to use for saving
            seed:           random number generator seed
            measure:        function to compare estimates to targets
            targets:        target states
            autosave:       if true then autosave
        
        """
        self.eng = None             
        self.autosave = autosave    
        self.out_value = []          
        if measure is None:
            measure = fidelity      
        self.measure = measure      
        self.targets = targets      
        self.setSeed(seed)

        super().__init__(hists, input_state, unitaries, P_j, bin_flag,
                         name)
        
        if hists is not None:
            self.bootstrap_log = []     
            self.est_rho = None 
            self.train_frac = train_frac  
            self.TrainParamters = None      # Training AnalysisParameter object
            self.trainingSample(self.train_frac)  # Calculate training set
            if self.targets is not None:
                self.targets = np.array(self.targets)
                assert(len(self.targets) == self.num_rhos or
                       self.num_rhos == 0)

    def setSeed(self, seed=None):
        """Sets seed of random number generator."""
        
        # if none, assign it so it will have a value to save
        if seed is None:  
            seed = np.random.randint(np.iinfo(np.int32).max)
            
        np.random.seed(seed)        
        self.seed = seed            
        
    def autobinMaxLike(self, nbins=None):
        """
        Bins all histograms based on TrainParamters instance.
        
        The binning rule is construced based on TrainParameters instance which
        is an AnlysisParameters object constructed in __init__ from 
        self.traininSample(). The binning rule is constructed from
        autobinParams().

        Args:
            nbins:  target number of bins
            
        Returns:
            mut_info_ratio: ratio of mutual information of new binning to
                            no binning along each axis.
                            
        """
        # use training data to bin when train_frac is nonzero        
        if self.train_frac > 0:
            # TrainParameters is a AnalysisParameters object
            self.TrainParameters.autobinParams(nbins) 
            self.bin_strat = self.TrainParameters.bin_strat
            
            # Training data may not have outer bounds correct, manually fix
            for i in range(self.dim_hist):
                self.TrainParameters.bin_bounds[i][0] = self.min_counts[i]
                self.TrainParameters.bin_bounds[i][-1] = self.max_counts[i]+1
                                               
            # Bin current instance of PartialMaxLike with bin_bounds
            self.rebinParams(self.TrainParameters.bin_bounds)
        else:
            super().autobinParams(nbins)
            
        return self.mutualInfoRatio()

    def trainingSample(self, train_frac=0.1):
        """
        Samples from references without replacement to generate training data.
        
        Randomly select train_frac of reference histogram data and use it to
        generate an instance of AnalysisParamters called TrainParameters. 
        
        Args:
            train_frac:     fraction of reference data to use for training
            
        """
        self.train_frac = train_frac  

        if (self.train_frac <= 0 or self.train_frac >= 1):
            return None

        TrainHists = []
        for i in range(self.ref_num):
            TrainHists.append(hg.Hist())
            ref_ind = self.ref_ind[i]
            train_trial = int(np.floor(self.train_frac*self.hists[ref_ind].trials))
            counts = hg.makeCounts(self.hists[ref_ind].raw_hist,
                                   self.hists[ref_ind].raw_hist_bins)
            # Samples random counts from raw histogram as training data
            train_ind = np.random.choice(self.hists[ref_ind].trials,
                                         train_trial, replace=False)
            # Pulls out training data
            hist_ind = np.setdiff1d(np.arange(
                                    self.hists[ref_ind].trials), train_ind)
            # Make both histograms (remake original with less counts)
            orig_hist = self.hists[ref_ind].hist       
            orig_bins = self.hists[ref_ind].bin_bounds
            TrainHists[i].makeFromCounts(counts[train_ind])
            self.hists[ref_ind].makeFromCounts(counts[hist_ind])
            TrainHists[i].rebin(orig_bins)
            self.hists[ref_ind].rebin(orig_bins)
            assert np.all(orig_hist ==
                          self.hists[ref_ind].hist + TrainHists[i].hist)

        # Return to binning for remaining reference histograms
        self.rebinParams()

        # Make training samples an AnalysisParameter object
        input_state = [np.array(self.input_state[self.ref_ind[k]]) 
                       for k in range(self.ref_num)]
        self.TrainParameters = AnalysisParameters(TrainHists, input_state,
                                                  self.unitaries[self.ref_ind], 
                                                  self.P_j)
      
    def tomography(self, rho_start=None, Q_start=None):
        """
        Applies iterative maximum likelihood quantum tomography.
        
        Creates .tom, a tomography object within PartialMaxLike, to do the 
        alternating maximum likelihood algorithm between state and measurement
        optimization.

        Args:
            rho_start: array of initial guess for each unknown state
                       to start tomography at
            Q_start:   initial guess for transition matrix
            
        Returns:
            out_value:  out_value of the "measure" between target states and 
                        the estimated states (est_rho)
        """
        initial_rho = [self.input_state[self.ref_ind][i]
                       for i in range(self.ref_num)]
        
        if rho_start is None:
            # Default initial rho is maximally mixed state
            initial_rho.extend([np.identity(self.dim)/self.dim
                                for k in range(self.probe_num)])
        else:
            # Convert rho_start list into full list for tomography
            initial_rho.extend([self.state_label[self.probe_ind][k]-1
                                for k in range(self.probe_num)])
        initial_rho = np.array(initial_rho)
        
        if Q_start is None:
            # Default initial Q is from linear inversion
            Q_start = self.initialQ()
            # If linear inversion failed, make simple initial point
            if Q_start is None:
                Q_start = np.ones((self.num_subs, self.bins)) / self.bins
        initial_Q = Q_start
        
        hists = self.convertListOfHistsToMatrix()
        self.tom = tm.Tomographer(initial_Q, initial_rho, hists, 
                                  self.state_label, self.P_ij)
        self.tom.iterativeTomography()
        self.est_rho = self.tom.est_rho_final
        self.est_Q = self.tom.Q
        
        # Calculate fidelities
        self.remeasure()
        self.autoSave()
        
        return self.out_value
    
    def remeasure(self, measure=None):
        """
        Recalculate value of measure.


        Recalculates inferred density matrices as well as all resamples
        from bootstrap.
        
        """
        if measure is not None:
            self.measure = measure
            
        # Calculate measure
        if self.targets is not None:
            self.out_value = []
            for i in range(self.num_rhos):
                self.out_value.append(self.measure(self.targets[i],
                                                  self.est_rho[i]))
            self.out_value = np.array(self.out_value)

        if self.bootstrap_log:
            self.bootstrap_value = [[self.measure(self.targets[k],
                                     self.bootstrap_log[i].est_rho_final[k])
                                     for k in range(self.num_rhos)]
                                     for i in range(self.boot_iters)]
            self.bootstrap_value = np.array(self.bootstrap_value)
            
    def bootstrap(self, method='parametric', iters=10):
        """
        Bootstrap to estimate variances for estimators.

        Args:
            method:     'nonparametric' or 'parametric'
            iters:      number of resamples

        Returns:
            bootstrap_log: tomographer list for each resample

        Bootstrap assumes targets for calling object same as bootstrap
        resamples
        
        """ 
        self.bootstrap_log = []   
        self.bootstrap_fidelities = []
        self.boot_iters = iters
        rho_start = None
        for i in range(self.boot_iters):
            if method == 'nonparametric':
                ResampledSuper = self.nonParametricResample()
                ResampledSuper.rebinParams(self.bin_bounds)  # use prebinned bins
            elif method == 'parametric':
                ResampledSuper = self.parametricResample()
            ResampledSuper.tomography(rho_start)
            self.bootstrap_log.append(ResampledSuper.tom)
            
    def parametricResample(self):
        """
        Resamples all histograms using estimated states and transition matrix.

        Note: this does not preserve the number of distinguishable ions nor
        can these histograms be rebinned to more bins.

        Returns: 
            ResampledSuper: PartialMaxLike instance for single bootstrap
                            resample
        
        """
        new_hists = []
        for i in range(self.num_hists):
            # use populations estimated from tomography to calculate
            # probability of observing of each "count"
            prob = self.tom.pops[i].dot(self.est_Q)
            new_hists.append(hg.Hist())
            hist = np.random.multinomial(self.hists[i].trials, prob)
            new_hists[-1].makeFromHist(hist, self.hists[i].bin_bounds)
            new_hists[-1].max_counts = self.hists[i].max_counts
            new_hists[-1].min_counts = self.hists[i].min_counts

        ResampledSuper = PartialMaxLike(new_hists, self.input_state,
                                        self.unitaries, self.P_j,
                                        train_frac=0, bin_flag=False,
                                        measure=self.measure,
                                        targets=self.targets, 
                                        autosave=False)
        # share matlab engine if it exists for expecation SDP
        # ResampledSuper.eng = self.startMatlabEng() \
        #                       if self.eng is not None else None
        if self.eng is not None:
            ResampledSuper.eng = self.startMatlabEng() 
        else:
            ResampledSuper.eng = None

        ResampledSuper.bin_bounds = self.bin_bounds
        ResampledSuper.mbins = self.mbins
        ResampledSuper.bins = self.bins
        
        return ResampledSuper
    
    def nonParametricResample(self):
        """
        Resamples all histograms based on data from all experiments.

        Returns: 
            ResampledSuper: PartialMaxLike instance for single bootstrap
                            resample
        
        """
        new_hists = []
        for i in range(self.num_hists):
            new_hists.append(hg.Hist())
            new_hists[-1].makeFromHist(self.hists[i].resample(),
                                       self.hists[i].rawhistbins)
            
        ResampledSuper = PartialMaxLike(new_hists, self.input_state,
                                        self.unitaries, self.P_j,
                                        train_frac=0, bin_flag=False,
                                        measure=self.measure,
                                        targets=self.targets, 
                                        autosave=False)
        # share matlab engine if it exists for expecation SDP
        #resampledSuper.eng = self.startMatlabEng() \
         #                    if self.eng is not None else None
        if self.eng is not None:
            ResampledSuper.eng = self.startMatlabEng() 
        else:
            ResampledSuper.eng = None
            
        return ResampledSuper
    
    def bootstrapAnalysis(self, measure=None):
        """
        Calculates statistics from bootstrap_log.
        
        Args:
            measure: function to compare targets and estimated states
        
        """
        if not self.bootstrap_log:
            print('No bootstrap to analyze?')
            return
        
        if measure is not None:
            self.measure = measure
        
        # Extracts final values of loglikelihood and measure
        self.loglike_final_est = self.tom.loglikelihood_list[-1]
        self.loglike_final_boot = [self.bootstrap_log[i].loglikelihood_list[-1]
                                   for i in range(self.boot_iters)]
        self.bootstrap_value = [[self.measure(self.targets[k],
                                 self.bootstrap_log[i].est_rho_final[k])
                                 for k in range(self.num_rhos)]
                                 for i in range(self.boot_iters)]
        self.bootstrap_value = np.array(self.bootstrap_value)
        
        # Calculates meaningful quantities from the bootstrap
        self.likelihood_percentile = percentileofscore(self.loglike_final_boot,
                                                       self.loglike_final_est)
        self.conf_int = np.array([2*self.out_value 
                                  - np.percentile(self.bootstrap_value, 97.5),
                                  2*self.out_value 
                                  - np.percentile(self.bootstrap_value, 2.5)])
        
        print('Bootstrap mean ', np.real(np.mean(self.bootstrap_value)))
        print('95 percent confidence interval: (%f, %f)' % \
              (np.real(2*self.out_value 
              - np.percentile(self.bootstrap_value, 97.5)),
              np.real(2*self.out_value 
              - np.percentile(self.bootstrap_value, 2.5))))
    
        self.autoSave()
           
    def inferExpectation(self, operators):
        """
        Estimate min and max epectation values for an array of operators.

        For each operator, a tuple of expectations are returned corresponding
        to the minimum expectation over possible density matrix
        consistent with the data, the expectation from the inferred density
        matrix, and the maximum. A semidefinite program calculates the minimum
        and maximum.

        If the bootstrap has been run then a tuple is returned for each
        resample

        Args:
            operators: array of hermitian matrices
            
        Returns:
            bound_list:     lower and upper bound for expectation value
            output_flag:    output flag from MATLAB call to YALMIP
            
        """
        operators = np.array(operators)
        if operators.ndim < 3:
            operators = np.expand_dims(operators, axis=0)
            
        bound_list = []
        flag_list = []
        log = self.makeFullLog()
        for tom in log:
            inferred_rho = tom.rho[[self.tom.probe_ind[0][k]
                                    for k in range(self.num_rhos)], ...]
            all_povms = tom.F[self.probe_ind]

            for p in range(self.num_rhos):
                # Pick out corresponding povm elements and "flatten"
                povms = self.svdPOVM(all_povms[self.probe_ind_data[p]])
                povms = all_povms[self.probe_ind_data[p]]
                povms = povms.reshape((-1, self.dim, self.dim))
                bounds, output_flag = measureOperator(operators[p], 
                                                      inferred_rho[p],povms, 
                                                      self.eng) 
                bound_list.append(np.array(bounds)) 
                flag_list.append(np.array(output_flag))
                
        return np.array(bound_list), np.array(flag_list) 
            
    def startMatlabEng(self):
        """Starts (or restarts) MATLAB engine."""
        
        if self.eng is None:
            print('Starting Matlab Engine...          ', end='')
            self.eng = matlab.engine.start_matlab()
            self.eng.addpath(
                self.eng.genpath(os.path.dirname(os.path.realpath(__file__))))
            print('[[DONE]]')
            
        return self.eng

    def killMatlabEng(self):
        """Kills MATLAB engine."""
        
        print('Killing Matlab Engine')
        self.eng.quit()
        self.eng = None
        
    def svdPOVM(self, povms=None):
        """
        Caclulates singular value decomp of POVM elements as vectors.

        Args:
            povms:  list of a set of povm elements
                    shape = (?, bins, dim, dim)
                    
        Returns: 
            row_basis: matrix containing condensed infromation on POVMs
                    
        """
        meas_dim = np.linalg.matrix_rank(self.P_ij[self.ref_ind,:,:,:].reshape(
                                         (-1,self.dim**2))) 
        povm_matrix = povms.reshape((-1, self.dim**2))
        u, s, vt = np.linalg.svd(povm_matrix)
        row_basis = vt[range(meas_dim)]
        
        return row_basis.reshape((-1, self.dim, self.dim))
    
    def simulate(self, stateSim=None, histSim=None, trials=5000,
                 train_frac=0.1, **simkwargs):
        """
        Generates simulated trapped-ion example described in paper.
        
        Uses function in simulate_tools.py to generate sample states, 
        unitaries, POVMs, and histograms that simulate trapped-ion experiments.
        Only allows for a single unknonw state.
        
        Args:
            stateSim:       function with arguements below
            rho:            array of initial state for reference experiments
                            and targets state for probing experiments
            Us:             array of known unitaries
            input_state:    array of initial states for reference experiments
                            and integer for probing states
            P_j:            array of underlying POVM elements
            histSim:        function that takes the above and below
                            arguments and returns:
            hists:          list of histogram objects
            trials:         number of trials for each hist (can be a list)
            train_frac:     fraction of trials used for training for each hist
            simkwargs:      keyword arguments to pass to state_sim and hist_sim
                            (assumes no inconsistent overlap in keyword 
                            arguments among those functions. Assumes these 
                            functions accept variable keyword arguments 
                            (**kwargs))
                            
        """
        if stateSim is None:
            if self.targets[0].size == 4:
                stateSim = st.oneQubit
            if self.targets[0].size == 16: 
                stateSim = st.twoQubitSym
        
        if stateSim is 'asym':
            stateSim = st.twoQubitAsym

        if histSim is None:
            histSim = st.oneDimPoissonHists
            
        if histSim is 'asym':
            histSim = st.nDimPoissonHists

        rho, Us, input_state, P_j = stateSim(**simkwargs)

        hists = histSim(rho, input_state, Us, P_j, trials, train_frac,
                         **simkwargs) 

        # Setup this object properly
        self.__init__(hists, input_state, Us, P_j, train_frac=train_frac,
                      name='simulated_'+str(datetime.date.today()),
                      measure=self.measure, targets=self.targets)

    def copy(self):
        """Deep copy an PartialMaxLike instance."""
        
        temp_eng = self.eng
        self.eng = None
        SE = copy.deepcopy(self)  # deepcopy can't copy matlab engines
        SE.eng = temp_eng
        self.eng = temp_eng
        
        return SE
    
    def makeFullLog(self):
        """
        Create list of all tomographer instaces.
        
        First element is tomographer instance for data, 
        all other elements are from bootstrap
        
        """
        full_log = [self.tom] + self.bootstrap_log
                   
        return full_log

    def autoSave(self):
        """
        Saves PartialMaxLike instance.
        
        This is to avoid bootstrap resampled objects from saving themselves
        and slowing the bootstrap down.

        """       
        if self.autosave is True:
            self.save('autosave_' + self.name)

    def save(self, filename):
        """Save all relevant information """
        
        temp = self.eng
        self.eng = None
        pickle.dump(self, open(filename + '.hist', 'wb'))
        self.eng = temp


# ----------------------------------------- #
### NECESSARY FUNCTIONS FOR ABOVE CLASSES ###
# ----------------------------------------- #

def measureOperator(O, inferred_rho, povms, eng):
    """ 
    Find upper and lower bound of observable for an estimated state. 
    
    This function is a wrapper for measureOperator.m (see documentation), which
    uses YALMIP in MATLAB to solve two semi-definite programs to find the lower
    and upper bounds of an expectation value.

    Args:
        O:              observable
        inferred_rho:   inferred state from iterative tomography procedure
        povms:          POVM
        eng:            MATLAB engine object
        
    Returns:
        bounds:      lower and upper bounds of expecitation value of observable
        output_flag: output flag from YALMIP describing stopping condition
        
   """
    # Convert to matlab variables
    # Need to make povm list a matlab list (roll the list axis)
    O = matlab.double(O.tolist(), is_complex=True)
    inferred_rho = matlab.double(inferred_rho.tolist(), is_complex=True)
    
    swapped = np.rollaxis(np.array(povms), 0, 3)
    povms = matlab.double(swapped.tolist(), is_complex=True)
    
    bounds, output_flag = eng.measureOperator(O, inferred_rho, povms, 
                                              nargout=2)
    
    return bounds[0], output_flag[0]

def binDists(unbn, bnds):
    """Bins 1D probability distributions (histograms).

    Args:
        unbn: probability distributions without binning
        bnds: new bin boundaries
        
    Returns:
        binned: binned probability distribution

    Assumes bins are contiguous
    
    """
    bins = np.size(bnds)-1
    bnds = bnds.astype(int)
    if unbn.ndim > 1: 
        binned = np.zeros([np.shape(unbn)[0], bins])
        for i in range(bins):
            binned[:,i] = np.sum(unbn[:,bnds[i]:bnds[i+1]], 1)
    elif unbn.ndim == 1:
        binned = np.zeros(bins)
        for i in range(bins):
            binned[i] = np.sum(unbn[bnds[i]:bnds[i+1]])
            
    return binned

def mutualInfo(pc, ps, p_cgs):
    """
    Computes mutual information between two random variables.

    Args:
        pc:    unconditional probablity of random variable C
        ps:    unconditional probablity of random variable S
        p_cgs: conditional probabilities of C given S
               each row is probability distribution of C given R_i
               
    Returns
        mutual_information: mutual information between pc and ps
        
    """
    mutual_information = 0
    for i in range(np.size(ps)):     # over states
        for j in range(np.size(pc)):    # over counts
            if p_cgs[i, j] > 0 and pc[j] > 0:
                mutual_information += p_cgs[i, j]*ps[i] * \
                                     np.log2(p_cgs[i, j]/pc[j])
    return mutual_information

def fidelity(rho, sigma):
    """Finds fidelity between density matrices rho and sigma."""
    
    # rho is a pure state
    if np.linalg.matrix_rank(rho) == 1:      
        F = np.trace(rho.dot(sigma))
    
    # sigma is a pure state
    elif np.linalg.matrix_rank(sigma) == 1:  
        F = np.trace(sigma.dot(rho))
    
    # Uhlmann's fidelity
    else:
        rho_half = linalg.sqrtm(rho)
        F = (np.trace(linalg.sqrtm(rho_half.dot(sigma).dot(rho_half))))**2
        
    return F


#def main():
#    SE = AnalysisSet(seed=0)
#    SE.simulate()
#    #SE.simulate(two_qubit_asym_bell_state, n_dimensional_poisson_hists)
#    SE.estimate_q_quick()
#    print(SE.autobin())
#    #SE.tomographyML()
#    #SE.bootstrap(iters=2)
#    #SE.killMatlabEng()
#    return SE
#
#
#if __name__ == "__main__":
#    main()

