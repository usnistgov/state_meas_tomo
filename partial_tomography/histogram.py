"""
Defines a histogram from a single experiment.

Assumptions:
    Raw histogram bins MUST be integer spaced (problems arise when adding
         histograms that did not at least come from integer spaced bins)
    Assumes integer bin boundaries
        Maybe assumes input histogram bin boundaries are integer spaced
        (i.e. don't prebin except for continuous measurements)
        Not sure if violating this assumption actually causes any problems
    All histograms are multidimensional for binning purposes
        histograms are flattened using (np.flatten() (row major)) for computing
        things like loglikelihoods, etc.
        Thus, bin boundaries are a list of bin boundaries for each dimension,
        even in 1D, yes sorry.

Problems:
    Adding histograms results in no binning. How do handle this properly?
    raw_hists have many zeros if loaded from some types of data. Not a problem
        but not efficient
    Not sure if code can handle general non-integer binbounds
    
Author = Adam Keith
    
"""
import copy

import numpy as np
from scipy.stats.contingency import margins


class Hist(object):
    """
    Collection of all counts from one experiment.
    
    An object that contains all trail outcomes from a single experiment. 

    Attributes:
        raw_hist:       Unbinned raw count list of each outcome
        raw_hist_bins:  Unbinned raw hist bin bounds (1,2,3,...)
        dim_hist:       number of possible outcomes in one trial?
        trials:         number of trials of experiment
        max_counts:     maximum counts (max value of outcome)
        min_counts:     minimum counts (min value of outcome)
        hist:           mutable array of histogram counts
        bin_bounds:     mutable array of boundaries
        bins:           mutable number of bins
    """

    def __init__(self, counts=None):
        """Generates histogram from experimental data."""
        
        # If no experiment make empty histogram
        self.raw_hist = []       
        self.raw_hist_bins = []   
        self.dim_hist = 1
        self.trials = 0         
        self.max_counts = []     
        self.min_counts = []     
        self.hist = []          
        self.bin_bounds = []     
        self.bins = []    
        
        if counts is not None:
            self.makeFromCounts(counts)

    def hist1D(self, raw=False):
        """Returns row major flattened histogram."""
        
        if raw is True:
            return self.raw_hist.flatten()
        return self.hist.flatten()

    def histND(self, flat):
        """Returns reshaped multidimensional histogram or distribution."""
        
        return flat.reshape(self.bins)

    def makeFromCounts(self, counts):
        """
        Generates Histogram from a list of counts.

        Make permanent unbinned histogram and setup mutable
        histogram to be binned from count array.

        Args:
            counts: list of observed values, may be list of tuples
            
        """
        counts = np.array(counts)
        counts = np.squeeze(counts)  # in case individual counts are lists

        if counts.ndim > 1:
            self.dim_hist = np.shape(counts)[1]  # length of each count tuple
        else:  # counts are 1D
            self.dim_hist = 1
            counts = np.expand_dims(counts, 1)  # need to add in dimension

        self.trials = np.shape(counts)[0]
        self.max_counts = np.amax(counts, 0)
        self.min_counts = np.amin(counts, 0)
        # initial bin boundaries are int spaced (assuming counts are int)
        ranges = np.transpose(np.array([self.min_counts,
                                        self.max_counts+1]))
        bins = self.max_counts+1 - self.min_counts
        self.raw_hist, self.raw_hist_bins = np.histogramdd(counts,
                                                           bins=bins,
                                                           range=ranges)                                            
        self.hist = copy.deepcopy(self.raw_hist)
        self.bin_bounds = copy.deepcopy(self.raw_hist_bins)
        self.bins = np.squeeze(bins)

    def binRawHist(self, new_bounds, self_flag=True):
        """
        Bins raw histograms.

        Args:
            new_bounds: new bin boundaries
                        1D array -> bin 1D histogram according to bin_bounds
                        list of 1D array -> Each sublist of newbinbounds is
                                            bin boundaries in multidimensional
                                            histogram
            self_flag:  if True set self.hist to new binned histogram
                        and self.bin_bounds to new_bounds
                        
        Returns:
            binned: binned histogram
                        
        """
        bins = []
        # sets bins as number of new bins in each dimension
        for n in range(self.dim_hist):
            bins.append(np.size(new_bounds[n])-1)

        # make new n-dimensional histogram which is copy of original
        binned = copy.deepcopy(self.raw_hist)
        hist_shape = np.asarray(self.raw_hist.shape)

        # bin each dimension individually
        for n in range(self.dim_hist):
            hist_shape[n] = bins[n]  # changes to number of new bins
            binned_temp = np.zeros(hist_shape)
            old_bounds = self.raw_hist_bins[n][:-1]
            full_ind1 = [slice(None)]*self.dim_hist  
            full_ind2 = [slice(None)]*self.dim_hist
            for i in range(bins[n]):
                # finds counts in old binning that fall  in this new bin
                ind = np.where(np.logical_and(new_bounds[n][i] <= old_bounds,
                                              old_bounds < new_bounds[n][i+1]))
                ind = np.asarray(ind).flatten() # gets rid of extra dimension
                
                # if ind is empty (no counts observed) move on
                if ind.size == 0:  
                    continue

                full_ind1[n] = ind  # full index on partial binned histogram
                full_ind2[n] = i    # full index on more binned histogram

                if binned[full_ind1].ndim < self.dim_hist:
                    # only one count for this bin in this dimension
                    binned_temp[full_ind2] = binned[full_ind1]
                else:
                    binned_temp[full_ind2] = np.sum(binned[full_ind1], axis=n)

            binned = binned_temp

        if self_flag:
            self.hist = binned
            self.bin_bounds = new_bounds
            self.bins = self.hist.shape
            
        return binned

    def rebin(self, new_bin_bounds=None, show_warnings=True):
        """
        Rebins individual histogram based on new bin bounds

        Args:
            new_bin_bounds: interpretation of this value determines how
                            how histogram is binned.
                            Default None -> No compression (integer binning 
                                            from min_count to max_count along 
                                            each dimension)
                            list of 1D array -> Each sublist of newbinbounds is
                                                bin boundaries in 
                                                multidimensional histogram
            show_warnings:  flag to print warnings about total counts

        Be careful, a particular binning may not capture all counts
        Warnings are printed if this happens.

        All binning happens on raw, multidimensional histograms
        
        """
        # when no new_bin_bounds, returns raw binning (integer bins)
        if new_bin_bounds is None:
            self.bin_bounds = copy.deepcopy(self.raw_hist_bins)
            self.hist = copy.deepcopy(self.raw_hist)
            self.bins = self.hist.shape
            return
        
        # when new_bin_bounds is a 1D array then applies to make new bins
        self.binRawHist(new_bin_bounds)
        
        if np.sum(self.hist) != self.trials and show_warnings:
            print('This rebinning doesn\'t capture all counts.')
            
        return self.bin_bounds

    def countsAlongDim(self, dim):
        """Find counts for marginal of unbinned histogram for dim."""
        
        counts = makeCounts(marginal(self.raw_hist, dim), 
                            [self.raw_hist_bins[dim]])
        np.random.shuffle(counts)
        
        return counts

    def makeFromHist(self, hist, bin_bounds):
        """
        Converts numpy histogram to Hist() instance.
        
        Assumes:
            -input histogram is unbinned
            -bin_bounds are integer spaced
        
        Args:
            hist:       numpy histogram 
            bin_bounds: bin boundaries of histogram (integer spaced)
        
        """
        self.raw_hist = copy.deepcopy(hist).astype(int)
        self.raw_hist_bins = copy.deepcopy(bin_bounds)
        self.hist = copy.deepcopy(hist).astype(int)
        self.bin_bounds = copy.deepcopy(bin_bounds)
        self.trials = np.sum(hist)
        self.dim_hist = hist.ndim
        self.findCountRange()

    def simPoisson(self, trials=50000, pops=[0.9995, 0.0005], mu=[2, 20],
                   make=True):
        """
        Simulates 1D counts with multiple Poisson distributions.

        Args:
            trials: the number of counts
            pops:   list of desired populations, sum(pops)=1
            mu:     list of means for multiple Poissons, same size as pops
            make:   use counts to make histogram for this object
            
        Returns:
            counts: histogram counts
        """
        counts = []
        trials_per_poisson = np.random.multinomial(trials, pops)
        for lam, N in zip(mu, trials_per_poisson):
            counts.extend(np.random.poisson(lam, N).tolist())
        np.random.shuffle(counts)
        if make:
            self.makeFromCounts(counts)
            
        return counts

    def simulateMulti(self, dim_hist=2, trials=50000,
                      state_pops=[0.4, 0.1, 0.1, 0.4], mu=[[2, 20], [3, 25]]):
        """
        Simulates counts for multi-species experiment using simPoisson().

        Args:
            trials:      the number of count tuples
            state_pops:  list of desired populations of multi-ion states
                         ((dark, dark), (bright, bright)), etc
            dim_hist:    number of dimensions (number of distinguishable ions)
            mu:          for each ion,
                          list of means for single ion states (Poissons)
                          
        Assumes ions only have two states (qubits) and binary ordering of
        states.
        
        """
        self.dim_hist = dim_hist
        self.trials = trials
        mu = np.array(mu)
        raw_counts = []
        trials_per_state = np.random.multinomial(trials, state_pops)
        for i in range(2**self.dim_hist):
            # get counts for each ion in that state
            state_counts = []
            for s in range(self.dim_hist):
                # convert state to mean count for that ion in this state
                # binary string to see if ion is dark or bright
                m = mu[s, ord(bin(i)[2:].zfill(self.dim_hist)[s])-48]
                # stack counts for each ion
                state_counts.append(self.simPoisson(trials_per_state[i], [1], 
                                                   [m], make=False))
            # break into tuples
            raw_counts.extend(np.transpose(state_counts).tolist())
        self.makeFromCounts(raw_counts)

    def resample(self, N=None):
        """
        Resamples (with replacement) from raw_hist. 
        
        Args:
            N: number of resampled trials
            
        """
        if N is None:
            N = self.trials
        hist = self.raw_hist.flatten()
        new_hist = np.random.multinomial(N, hist/N)
        
        return self.histND(new_hist)

    def findCountRange(self):
        """Find max and min values along each dimension."""
        
        self.max_counts = np.zeros(self.dim_hist)
        self.min_counts = np.zeros(self.dim_hist)
        for n in range(self.dim_hist):
            # I could make this tighter, by check last nonzero item in marginal
            self.max_counts[n] = self.raw_hist_bins[n][-2]  # second to last item
            self.min_counts[n] = self.raw_hist_bins[n][0]

    def trialsAboveCutoff(self, cutoff):
        """Returns number of trials with counts above cutoff (inclusive)."""
        
        # needs to be fixed for multiions
        unbnbnds = self.raw_hist_bins
        ind = np.where(unbnbnds == cutoff)
        
        return np.sum(self.raw_hist[ind:])

    def trialsBelowCutoff(self, cutoff):
        """Returns number of trials with counts below cutoff (inclusive)."""
        
        return (self.trials - self.trialsAboveCutoff(cutoff+1))

    def __add__(self, other):
        """Adds Hist() instances."""
        
        added_hist = copy.deepcopy(self)
        added_hist += other
        
        return added_hist

    def __iadd__(self, other):
        """Adds assign Histograms.

        Concatenate count lists and recreate histogram

        Note: This method removes previous binning on self
        
        """
        # Check if same type
        if isinstance(self, Hist) and isinstance(other, Hist):
            # Check if both hists have same dim_hist
            if self.dim_hist == other.dim_hist:
                # Generate counts for each histogram, concatenate
                # and recreate histogram
                c1 = makeCounts(self.raw_hist, self.raw_hist_bins)
                c2 = makeCounts(other.raw_hist, other.raw_hist_bins)                
                counts = np.concatenate((c1, c2))
                #counts = np.concatenate((c1, c2), axis=0)
                self.makeFromCounts(counts)
            else:
                raise TypeError('Attempted to add histograms with '
                                'different histogram dimension')
        else:   # if other is array or single number, add it to each bin
            self.hist += other
            
        return self

    def __radd__(self, other):
        """Used for sum() function."""
        
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def save(self, fname='histogram_counts.txt'):
        """Convert raw_hist to counts and save as a text document"""
        c = makeCounts(self.raw_hist, self.raw_hist_bins)
        np.random.shuffle(c)
        np.savetxt(fname, c, '%i')


def marginal(dist, dim):
    """
    Compute marginal of distribution dist along axis dim.
    
    Note:
        Computes all marginals and returns the one asked, so might be slow
    
    """
    ms = margins(dist)   # compute all marginals
    return np.squeeze(ms[dim])   # pick out the desired one


def makeCounts(hist, bin_bounds):
    """
    Make a count list from histogram.
    
    Note:
        The count list is ordered and unshuffled
        May not work for multi dimensional histograms
        
    """
    hist = np.array(hist)
    counts = []

    ind = np.array(np.unravel_index(np.arange(hist.size), hist.shape))
    hist = hist.flatten()
    for c in range(hist.size):
        counts.extend(np.tile(ind[..., c], (int(hist[c]), 1)).tolist()) 
    #np.random.shuffle(counts)
    counts = np.squeeze(np.array(counts))
    # Map to actual values using binbounds
    dim_hist = len(bin_bounds)  # or len(hist.shape)
    if dim_hist == 1:
        counts = np.expand_dims(counts, 1)  # need to add in dimension
    for n in range(dim_hist):
        counts[:, n] = bin_bounds[n][counts[:, n]]
        
    return np.squeeze(counts)  
