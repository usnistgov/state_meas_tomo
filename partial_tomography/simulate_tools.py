"""
Functions for simulating analysis on two qubits.
 
Assumes |down> = |0>, |up> = |1> 
        |dark>        |bright>    

Author = Adam Keith

"""
import numpy as np
from histogram import Hist


def generateExpectedCounts(states, background, counts_per_state):
    """
    Generates linear list of expected counts for each state.

    Args:
        states:           number of states
        background:       background counts (only for darkest state)
        counts_per_state: number of counts per state
        
    Returns:
        expected_counts: average counts for counting statists
        
    """
    expected_counts = np.arange(0, states*counts_per_state, counts_per_state)
    expected_counts[0] = background
                   
    return expected_counts


def fiducial(dim=4):
    """
    Makes fiducial state.
    
    Assumes last state in binary ordering is prepared. 
    Called the "bright state" 
    
    Args:
        dim: Dimension of hilbert space
        
    Returns: 
        rho: fiducial state
        
    """
    rho = np.zeros((dim, dim))
    rho = rho+0j
    rho[-1, -1] = 1.0
       
    return rho


def makeRho(dim=4, prob=1):
    """
    Make bell state density matrix + maximally mixed state of dimension dim.
    
    Only applies for d = 2,4,8 using equation:
    rho = prob*(bell state density matrix) + (1-prob)*(maximally mixed state)
    
    Args: 
        dim:  dimension of hilbert space
        prob: defines mixture with maximally mixed state by equation above
        
    Returns: 
        rho: target state 
    
    """
    if dim == 2:
        state_vector = (1/np.sqrt(2))*np.array([1, 1])
    if dim == 4:
        # Standard 2 qubit bell state
        state_vector = (1/np.sqrt(2))*np.array([1,0,0,1])
    elif dim == 8:
        # W State for three ions
        state_vector = (1/np.sqrt(3))*np.array([0, 1, 1, 0, 1, 0, 0, 0])
    ideal = np.outer(state_vector, state_vector)
    
    rho = prob*ideal + (1-prob)*np.identity(dim)/dim
                       
    return rho + 0j


def pulseSequence(theta, phi, N=2):
    """
    Makes N qubit unitary as a product of rotation unitaries.
    
    If theta,phi are arrays (with equal length), make unitary a product of 
    rotation matrices with those params. First theta is right most unitary.

    Args:
        theta:   angle to rotate
        phi:     list of phase angles of rotation unitaries
        N:       number of qubits
           
    Returns:
        U: Unitary that is a combination of pulse sequence
           
    """
    theta = np.array(theta)
    phi = np.array(phi)
    assert theta.size == phi.size
    U = np.eye(2**N)
    for i in range(theta.size):
        t = theta[i]
        p = phi[i]
        R = nistRotation(t, p)
        temp = R
        for i in range(N-1):
            temp = np.kron(temp, R)
        U = temp.dot(U)
        
    return U


def nistRotation(theta=np.pi/2, phi=0):
    """Constructs NIST standard rotation matrix."""
    
    # Assumes |0>=|dark>, |1>=|bright>
    R = np.array([[np.cos(1/2*theta),
                  -1j*np.exp(-1j*phi)*np.sin(1/2*theta)],
                  [-1j*np.exp(1j*phi)*np.sin(1/2*theta),
                  np.cos(1/2*theta)]])
    
    return R


def rotationError(theta_eps=0):
    """Small rotation between two states. Just rotate around phi=0."""
    
    R = np.array([[np.cos(1/2*theta_eps), -1j*np.sin(1/2*theta_eps)],
                  [-1j*np.sin(1/2*theta_eps), np.cos(1/2*theta_eps)]])
    
    return R


def symStateMap(N=2):
    """
    Make state map for symmetric qubits. 
    
    All the subspaces with the same Hamming weight (number of excitations)
    are indistinguishable from each other. This is the case considered as the
    example in the paper.

    Args:
        N: number of qubits

    Output:
        f:          state_map (underlying POVM elements)
        num_sub:    number of subspaces is N + 1
        
    """
    num_subs = N+1
    f = np.zeros((2**N, num_subs))
    for i in range(2**N):
        for j in range(num_subs):
            # find all states for same hamming weight j
            if (bin(i).count("1") == j):
                f[i, j] = 1
                 
    return f, num_subs


def oneQubit(fid=0.99):
    """
    Generates single-qubit known unitaries, reference, and probing experiments.
    
    Args:
        fid:  fidelity of state preparation, used in makeRho

    Returns: 
        rho:         array of fiducial state for reference experiment and 
                     target states for probing experiments
        Us:          knonw unitaries
        input_state: array of fiducial state for reference experiment and 1 for 
                     probing experiment
        P_j:         underlying POVM elements
    
    """
    
    rho = []
    Us = []
    input_state = []
    N = 1
    dim = 2**N
    
    # projectors
    state_map, num_sub = symStateMap(N)
    P_j = np.zeros((num_sub, dim, dim))
    for j in range(num_sub):
        P_j[j, :, :] = np.diag(state_map[:, j])
        
    # state perparations
    fid_state = fiducial(dim=dim)
    target_state = makeRho(dim=dim, prob=fid)
    
    # known unitaries
    known_theta = np.pi*np.array([0,1/2])
    known_phases = np.pi*np.array([0,1/2])
    num_known = known_phases.size
    

    # Reference experiments
    for i in range(num_known):
        rho.append(fid_state)
        Us.append(nistRotation(known_theta[i], known_phases[i]))
        input_state.append(rho[-1])
    
    # Probe experiments
    for i in range(num_known):
        rho.append(target_state)
        Us.append(nistRotation(known_theta[i], known_phases[i]))
        input_state.append(1)
        
    return (rho, Us, input_state, P_j)
    

def twoQubitSym(fid=0.99):
    """
    Generates two-qubit parameters for symmetric measruements.
    
    All the subspaces with the same Hamming weight (number of excitations)
    are indistinguishable from each other. This is the case considered as the
    example in the paper. Uses oneDimPoissonHists to generate corresponding
    histograms.
    
    Args:
        fid:  fidelity of state preparation, used in makeRho

    Returns: 
        rho:         array of fiducial state for reference experiment and 
                     target states for probing experiments
        Us:          knonw unitaries
        input_state: array of fiducial state for reference experiment and 1 for 
                     probing experiment
        P_j:         underlying POVM elements
    
    """
    
    rho = []
    Us = []
    input_state = []
    N = 2
    dim = 2**N
    
    # projectors
    state_map, num_sub = symStateMap(N)
    P_j = np.zeros((num_sub, dim, dim))
    for j in range(num_sub):
        P_j[j, :, :] = np.diag(state_map[:, j])
        
    # state preparations
    fid_state = fiducial(dim=dim) 
    target_state = makeRho(dim=dim, prob=fid)

    # known unitary processes
    known_theta = np.pi*np.array([0,1/2,1/2,1])
    known_phases = np.pi*np.array([0,0,1/2,0])
    num_known = known_phases.size

    # reference experiments
    for i in range(num_known):
        rho.append(fid_state)
        Rtmp = nistRotation(known_theta[i], known_phases[i])
        Us.append(np.kron(Rtmp, Rtmp))
        input_state.append(rho[-1])
    
    # probe experiments (uses same known unitaries but could be different)
    for i in range(num_known):
        rho.append(target_state)
        Rtmp = nistRotation(known_theta[i], known_phases[i])
        Us.append(np.kron(Rtmp, Rtmp))
        input_state.append(1)
        
    return (rho, Us, input_state, P_j)


def oneDimPoissonHists(rho, input_state, Us, P_j, trials, train_frac, mu=None,
                       **simkwargs):
    """
    Generates 1D histograms for simulated examples.
    
    Args:
        rho:         array of fiducial state for reference experiment and 
                     target states for probing experiments
        input_state: array of fiducial state for reference experiment and 1 for 
                     probing experiment
        Us:          knonw unitaries
        P_j:         underlying POVM elements
        trials:      number of trials per experiment
        train_frac:  fraction of reference histograms used for trainind data
        mu:          average number of photon counts for each underlying POVM
                     outcome
        **simkwargs: keyword arguements for properties of state prep (fid)
                     
    Returns:
        hists: histograms for reference and probing experiments
    
    """
    Us = np.array(Us)
    rho = np.array(rho)
    dim = rho.shape[-1]
    P_j = np.array(P_j)
    assert Us.shape[0] == rho.shape[0]
    num_sub = P_j.shape[0]      # number of subspaces
    num_total = Us.shape[0]     # total number of experiments

    # engineered POVM elements
    P_ij = np.zeros((num_total, num_sub, dim, dim))+0j
    for i in range(num_total):
        for j in range(num_sub):
            P_ij[i,j,:,:] = (Us[i].conj().T).dot(P_j[j]).dot(Us[i])

    pops = np.real(np.trace(np.einsum('iak,ijkb->ijab', rho, P_ij), 
                            axis1=-1, axis2=-2))

    # Reduce number of trials by 10% of trials for binnning
    trials = trials*np.ones(num_total) 
    ref_ind = np.where(list(np.array(input_state[i]).size > 1 
                            for i in range(num_total)))
    trials[ref_ind] = (1/(1-train_frac))*trials[ref_ind] 
    
    if mu is None:
        # in High-fidelity universal gate-set paper it says 30 photons
        mu = generateExpectedCounts(num_sub, 2, 20) 

    hists = []
    for k in range(num_total):
        hists.append(Hist())
        hists[-1].simPoisson(trials=trials[k], pops=pops[k], mu=mu)
        
    return hists


def twoQubitAsym(fid=0.99):
    """
    Generates two-qubit parameters for asymetric measurements.
    
    State preparations, unitaries and underlying POVM elements when each
    ion can be individually addressed. Uses nDimPoissonHists to generate 
    corresponding histograms.
    
    Args:
        fid:  fidelity of state preparation, used in makeRho

    Returns: 
        rho:         array of fiducial state for reference experiment and 
                     target states for probing experiments
        Us:          knonw unitaries
        input_state: array of fiducial state for reference experiment and 1 for 
                     probing experiment
        P_j:         underlying POVM elements
    
    """
    rho = []
    Us = []
    input_state = []
    N = 2
    dim = 2**N
    num_sub = dim
    
    # projetors
    state_map = np.eye(dim)
    P_j = np.zeros((num_sub, dim, dim))
    for j in range(num_sub):
        P_j[j, :, :] = np.diag(state_map[:, j])
 
     # state preparations
    fid_state = fiducial(dim=dim) 
    target_state = makeRho(dim=dim, prob=fid)

    # add in fiducial error
    eps = 0
    error = np.diag([eps**2, eps*(1-eps), eps*(1-eps),
                     eps**2-2*eps])
    fid_state = fiducial(dim=dim) + error

    # reference experiments (symmetric rotations)
    ref_theta = np.pi*np.array([0,1])
    ref_phases = np.pi*np.array([0,0])
    num_ref = ref_phases.size
    for i in range(num_ref):
        Rtmp1 = nistRotation(ref_theta[i], ref_phases[i])
        for j in range(num_ref):
            rho.append(fid_state)
            Rtmp2 = nistRotation(ref_theta[j], ref_phases[j])
            Us.append(np.kron(Rtmp1, Rtmp2))
            input_state.append(rho[-1])
    
    # probe experiments 
    probe_theta = np.pi*np.array([0,1/2,1/2])
    probe_phases = np.pi*np.array([0,0,1/2])
    num_probe = probe_phases.size
    for i in range(num_probe):
        rho.append(target_state)
        Rtmp = nistRotation(probe_theta[i], probe_phases[i])
        Us.append(np.kron(Rtmp, Rtmp))
        input_state.append(1)
        
    return (rho, Us, input_state, P_j)


def nDimPoissonHists(rho, input_state, Us, P_j, trials, train_frac, mu=None,
                     **simkwargs):
    """
    Generates n-dimensional histograms for simulated examples.
    
    Args:
        rho:         array of fiducial state for reference experiment and 
                     target states for probing experiments
        input_state: array of fiducial state for reference experiment and 1 for 
                     probing experiment
        Us:          knonw unitaries
        P_j:         underlying POVM elements
        trials:      number of trials per experiment
        train_frac:  fraction of reference histograms used for trainind data
        mu:          average number of photon counts for each underlying POVM
                     outcome
        **simkwargs: keyword arguements for properties of state prep (fid)
                     
    Returns:
        hists: histograms for reference and probing experiments
    
    """
    
    Us = np.array(Us)
    rho = np.array(rho)
    assert Us.shape[0] == rho.shape[0]
    dim = P_j.shape[2]
    num_sub = P_j.shape[1]  # number of subspaces
    num_total = Us.shape[0]  # number of histograms
    
    # rotated projectors
    P_ij = np.zeros((num_total, num_sub, dim, dim))+0j
    for i in range(num_total):
        for j in range(num_sub):
            P_ij[i,j,:,:] = (Us[i].conj().T).dot(P_j[j]).dot(Us[i])

    pops = np.real(np.trace(np.einsum('iak,ijkb->ijab', rho, P_ij), 
                            axis1=-1, axis2=-2))

    # Reduce number of trials by 10% of trials for binnning
    trials = trials*np.ones(num_total) # old code
    ref_ind = np.where(list(np.array(input_state[i]).size > 1 
                            for i in range(num_total)))
    trials[ref_ind] = (1/(1-train_frac))*trials[ref_ind] 
    
    if mu is None:
        mu = generateExpectedCounts(num_sub, 2, 20)
        mu = np.tile(mu, (2, 1))

    hists = []
    for k in range(num_total):
        hists.append(Hist())
        hists[-1].simulateMulti(dim_hist=2, trials=trials[k],
                                state_pops=pops[k], mu=[[2, 20],[2, 20]])
    return hists
