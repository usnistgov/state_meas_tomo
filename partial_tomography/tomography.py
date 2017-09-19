"""
Performs partial maximum likelihood estimation.

Most of this code translated from Scott Glancy's Matlab tomography project.
Some features from that code are missing here.

Problems:
    -Python solvers can't handle Nans or Inf in gradient? Start with mixed Q
    -Print warnings for max iterations
    -Failure at large number of bins
    -Large memory use because each experiment $i$ has its own rho
    
Author = Adam Keith
    
"""

import copy

import numpy as np
from scipy import optimize as opt


class Tomographer(object):
    """
    Performs partial quantum state tomography.
    
    Attributes:
        dim:            dimension of Hilbert space
        num_hist:       total number of histograms
        probe_ind:      indeces of probe experiments
        probe_trails:   total number of trials for all probing experiments
        P_ij:           engineered underlying POVM elements
        F:              POVM describing full measurement 
                        (underlying POVM + Markov process)
        num_sub:        number of underlying POVM elements
        bins:           number of bins (same for all histograms)
        hists:          arrray of histograms
        state_labels:   label of histogram type (0=reference, int=probing)
        iterations_rho: array of number of iterations for RrhoR algorithm for 
                        each Q iteration
        iterations_Q:   number of iterations of Q optimization algorithm
        max_iters_rho:  maximum number of RrhorR algorithm
        max_iters_Q:    maximum number of Q algorithm
        stop_Q:         stopping threshold for Q based on max difference bound
        stop_rho:       stopping threshold for RrhoR based on max difference 
                        bound
        stop_full:      sum of stop_Q and stop_rho
        R:              matrix used in RrhoR iterative algorithm
        rho:            current estimates of states
        Q:              current estimate of Q
        pops:           probability of each outcome from current rho

    
    """
    def __init__(self, initial_Q, initial_rho, hists, state_label, P_ij):
        """
        Initialize tomographer with supplied paramters.

        Args:
            initial_Q:   initial estimate of transition matrix
            initial_rho: initial estimate of unknown states 
            hists:       array of 1D histograms (first index labels setting)
            state_label: distinguishes between references and data histograms
            state_map:   maps computational basis states to subspaces
            
        """
        self.dim = P_ij.shape[-1]            
        self.num_hists = P_ij.shape[0]
        self.num_subs = P_ij.shape[1]
        self.bins = hists[0].size  
        
        self.P_ij = P_ij  
        self.hists = np.array(hists)         
        self.state_label = state_label       
        self.iterations_Q = 0         
        self.iterations_rho = []
        self.max_iters_rho = 10000        
        self.max_iters_Q = 10000     
        self.stop_Q = 0.25 
        self.stop_rho = 0.3 
        self.stop_full = self.stop_Q + self.stop_rho
        self.rho = initial_rho 
        self.Q = initial_Q
        
        self.selectSubsets()          
        self.modelIndepLoglike()     
        self.updatePops()
        self.updatePOVM()
        self.makeR()
        
    def selectSubsets(self):
        """Finds indeces of histograms and states to be inferred."""
        
        probe_labels = np.unique(self.state_label)  # labels for groups of hists
        probe_labels = probe_labels[probe_labels!=0]  # only want data labels
        self.num_rhos = probe_labels.size           # number of rho to estimate
        self.probe_ind = [np.where(self.state_label == probe_labels[k])[0]
                          for k in range(self.num_rhos)]
        self.probe_trials = [np.sum(self.hists[self.probe_ind[k]])
                             for k in range(self.num_rhos)]
        
    def modelIndepLoglike(self):
        """Computes loglikelihood for empirical frequencies."""
        
        np.seterr(all='ignore')
        L = self.hists*np.log(self.hists/np.sum(self.hists, 1)[:, np.newaxis])
        L = np.sum(L[np.nonzero(self.hists)])  # drop terms that have no counts
        self.loglike_model_indepedent = L
        
    def updatePops(self):
        """Calculates probability of each engineered underlying outcomes."""
        
        self.pops = np.real(np.trace(np.einsum('iak,ijkb->ijab', self.rho,
                                               self.P_ij), axis1=-1, axis2=-2))

    def updatePOVM(self):
        """Calculates POVMs for underlying + Markov process."""
        
        self.F = np.sum(self.Q[np.newaxis, ..., np.newaxis, np.newaxis] *
                        self.P_ij[:, :, np.newaxis, ...], 1)
        
    def makeR(self):
        """Makes R matrix for RrhoR for each rho."""
        
        R = []
        HIC = self.hists/self.pops.dot(self.Q)
        HIC[self.hists == 0] = 0  # erase infs and nans if histogram is 0
        for ind in self.probe_ind:  # for each rho to estimate
            R.append(np.sum(HIC[ind][..., np.newaxis, np.newaxis]*
                     self.F[ind], (0, 1)))
        self.R = np.array(R)
    
    def iterativeTomography(self):
        """
        Performs Max Like tomography for Q and rho.
        
        Alternates between RrhoR algorithm for rho and nonlinear convex 
        optimzation of Q until either max iterations_Q is reached or both
        stopping thresholds are reached.
        
        Returns: 
            stoppingCriteraQ:   stoping conditon for Q (if no rhos specified)
            final_stop:         sum of stopping conditions for rho and Q
        
        """
        self.loglikelihood_list = []
        
        if self.num_rhos == 0:
            self.loglikelihood_list.append(self.estimateQ())
            self.est_rho_final = 0 
            return self.stoppingCriteriaQ()
        
        self.iterations_rho.append(0)
        thresh_cond = True
        iter_cond = True
        diff_cond_rho = True
        while thresh_cond and iter_cond:
            # runs RrhoR algorithm
            diff_cond_rho = True
            iter_cond_rho = True
            diff_cond_Q = True
            while diff_cond_rho and iter_cond_rho:
                self.loglikelihood_list.append(self.estimateRho())
                if self.stoppingCriteriaRho() < self.stop_rho:
                    diff_cond_rho = False
                if self.iterations_rho[-1] >= self.max_iters_rho:
                    iter_cond_rho = False
             
            # runs Q algorithm
            self.loglikelihood_list.append(self.estimateQ()) 
            if self.stoppingCriteriaQ() < self.stop_Q:
                diff_cond_Q = False
                
            if self.iterations_Q >= self.max_iters_Q:
                iter_cond = False
                
            # checks each stopping condition individually
            if diff_cond_Q == False and diff_cond_rho == False:
                thresh_cond = False
            self.iterations_rho.append(0) 
            
        # checks sum of stopping condition
        self.final_stop = diff_cond_Q + diff_cond_rho
        if self.final_stop >= self.stop_full:
            print('Warning: total stopping criteria not met', 
                  self.iterations_Q,
                  np.max(self.iterations_rho))

        self.updatePops()
        self.updatePOVM()
        self.est_rho_final = self.rho[[self.probe_ind[k][0]
                                        for k in range(self.num_rhos)]]
        
        return self.final_stop

    def estimateRho(self):
        """
        Advances by an RrhoR iteration for each state.
        
        Returns: 
            loglikelihood: total loglikelihood after iteration
            
        """
        self.iterations_rho[-1] += 1
        self.makeR()
        for k in range(self.num_rhos):
            rho = self.R[k].dot(
                  self.rho[self.probe_ind[k][0]]).dot(self.R[k])
            rho = rho/np.trace(rho)
            rho = self.fixRho(rho)

            # update all "rhos" for this density matrix
            for i in self.probe_ind[k]:
                self.rho[i] = rho
        self.updatePops()
        
        return self.loglikelihood()
    
    def estimateQ(self):
        """
        Optimizes Q with nonlinear convex optimization.
        
        Uses scipy optimize package to find the Q that minimize the measurement
        loglikelihood function for fixed rho. 
        
        Returns:
            loglikelihood: total loglikelihood after optimization
        
        """
        self.iterations_Q += 1
        Q0 = self.Q.flatten()
        # Box constraints - all elements are between 0 and 1
        bounds = [(0, 1) for i in range(self.Q.size)]

        # Construct constraints (sum of probabilities add to 1)
        constraints = []
        for i in range(self.num_subs):
            def counstraintsQ(x, ind=i):
                temp = x.reshape((self.num_subs, self.bins))
                return np.sum(temp[ind])-1
            constraints.append({'type': 'eq', 'fun': counstraintsQ})

        res = opt.minimize(Tomographer.loglikelihoodQ, Q0, (self,),
                           method='SLSQP', jac=True,
                           bounds=bounds,
                           constraints=constraints,
                           tol=1.0e-9, 
                           options={'maxiter': 1000, 'disp': False})

        self.Q = res.x.reshape((self.num_subs, self.bins))
        self.Q = self.Q/np.sum(self.Q, 1)[:, np.newaxis]
        self.updatePOVM()
        
        return self.loglikelihood()    
    
    def loglikelihoodQ(Q_var, CurrentTom):
        """
        Computes negative loglikelihood with fixed rho.
        
        Q_var is the free parameter of the negative loglikelihood function,
        usedin the optimization over Q in estimateQ. This method is called
        with CurrentTom so that it can be used as the optimization function
        in estimateQ.
        
        Args:
            Q_var:      Value of transition matrix
            CurrentTom: current version of Tomographer object
        
        Returns: 
            neg_loglike_Q:  negative loglikelihood with fixed rhos
            neg_grad:       gradeint of negative loglikelihood
            
        """
        # reshapes Q to be matrix not vector (as is required for optimization)
        Q_var = Q_var.reshape((CurrentTom.num_subs, CurrentTom.bins))
        
        # Calculates loglike for constant rhos (and therefore pops)
        prob = CurrentTom.pops.dot(Q_var) 
        L = CurrentTom.hists*np.log(prob)
        L = np.sum(L[np.nonzero(CurrentTom.hists)]) 
        neg_loglike_Q = -(L - CurrentTom.loglike_model_indepedent)
        
        # Calculates gradient
        HIC = CurrentTom.hists/prob
        HIC[CurrentTom.hists == 0] = 0  # erase infs and nans if histogram is 0
        grad = np.transpose(CurrentTom.pops).dot(HIC) 
        neg_grad = -grad.flatten()
        
        
        return (neg_loglike_Q, neg_grad) 

    def loglikelihood(self):
        """
        Computes the total loglikelihood.
        
        Note: this is actually what is called the loglikelihood ratio in the
        paper but since the ratio is respect to the model independent, 
        loglikelihood, which is constant, optimizing this quantity is 
        equivalent.
        
        Returns:
            loglikelihood_ratio: difference between the loglikelihood at 
                                 current iteration and the model indpendent
                                 loglikelihood
                                 
        """
        L = self.hists*np.log(self.pops.dot(self.Q))
        L = np.sum(L[np.nonzero(self.hists)])  # drop terms that have no counts
        loglikelihood_ratio = L - self.loglike_model_indepedent
        
        return loglikelihood_ratio

    def stoppingCriteriaRho(self):
        """
        Calculates stopping criteria for R rho R algorithm.
        
        Returns
            stop_rho: bound on maximum amount of improvement for RrhoR
        
        """  
        stop = []
        for k in range(self.num_rhos):
            stop.append(np.real(np.max(np.linalg.eigvals(self.R[k])) - 
                                self.probe_trials[k]))
        stop_rho = np.max(stop)
        return stop_rho
        
    def stoppingCriteriaQ(self):
        """
        Calculates the stopping criteria for Q optimization.
        
        Uses a linear program to estimate the maximum value of the difference
        bound for the loglikelihood optimization over Q.
        
        Returns:
            stop_Q: bound on maximum amount of improvement for Q optimization
        
        """
        stop_Q = []
        # Calculates constraints
        Aeq = np.zeros((self.num_subs, self.Q.size))
        for j in range(self.num_subs):
            Aeq[j,j*self.bins:(j+1)*self.bins] = np.ones(self.bins)
        beq = np.ones(self.num_subs)
        bounds = [(0, 1) for i in range(self.Q.size)]
        
        L, grad = Tomographer.loglikelihoodQ(self.Q.flatten(), self)
        try:
            res = opt.linprog(grad, A_ub=None, b_ub=None, A_eq=Aeq, b_eq=beq,
                              bounds=bounds, method='simplex', callback=None,
                              options=None)
            stop_Q = -grad.dot(res.x-self.Q.flatten())
        except:
            print('Linear program failed?')
            stop_Q = self.stop_Q+1
            
        return stop_Q
    
    def fixRho(self, rho):
        """Makes rho a quantum state (positive semidefinte with trace one)."""
        
        # makes rho hermitian
        rho = (rho.conj().T + rho)/2  
        
        # Forces positivity
        try:
            min_eig = np.min(np.linalg.eigvalsh(rho))
        except:
            print(rho)
        if min_eig < 0:
            rho = rho - min_eig*np.eye(self.dim)
            rho = rho/(1-min_eig*self.dim)

        # Forces trace 1
        rho = rho/np.trace(rho)
        
        return rho
    
    def checkPOVM(self):
        """Asserts that POVM is physical (positive semidefinte elemnts)."""

        # Asserts POVM elements are hermitian
        assert np.all(np.transpose(self.F.conj(), axes=(0, 1, 3, 2)) == self.F)

        # Asserts POVM is positive semidefinite
        assert np.all(np.array([np.linalg.eigvalsh(
                      self.F.reshape((-1, self.dim, self.dim))[i, ...])
                      for i in range(self.bins*self.num_hists)]) >
                      (-100*np.finfo(float).eps))

        # Asserts each POVM sums to identity
        assert np.all(np.around(np.sum(self.F, 1), decimals=12) ==
                      np.eye(self.dim))
    
    def copy(self):
        """Makes deep copy a tomographer instance."""
        
        return copy.deepcopy(self)
