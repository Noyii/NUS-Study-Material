""" CS5340 Lab 3: Hidden Markov Models
See accompanying PDF for instructions.

Name: Niharika Shrivastava
Email: e0954756@u.nus.edu
Student ID: A0254355A
"""
import numpy as np
import scipy.stats
from scipy.special import softmax
from sklearn.cluster import KMeans


def initialize(n_states, x):
    """Initializes starting value for initial state distribution pi
    and state transition matrix A.

    A and pi are initialized with random starting values which satisfies the
    summation and non-negativity constraints.
    """
    seed = 5340
    np.random.seed(seed)

    pi = np.random.random(n_states)
    A = np.random.random([n_states, n_states])

    # We use softmax to satisify the summation constraints. Since the random
    # values are small and similar in magnitude, the resulting values are close
    # to a uniform distribution with small jitter
    pi = softmax(pi)
    A = softmax(A, axis=-1)

    # Gaussian Observation model parameters
    # We use k-means clustering to initalize the parameters.
    x_cat = np.concatenate(x, axis=0)
    kmeans = KMeans(n_clusters=n_states, random_state=seed).fit(x_cat[:, None])
    mu = kmeans.cluster_centers_[:, 0]
    std = np.array([np.std(x_cat[kmeans.labels_ == l]) for l in range(n_states)])
    phi = {'mu': mu, 'sigma': std}

    return pi, A, phi


"""E-step"""
def e_step(x_list, pi, A, phi):
    """ E-step: Compute posterior distribution of the latent variables,
    p(Z|X, theta_old). Specifically, we compute
      1) gamma(z_n): Marginal posterior distribution, and
      2) xi(z_n-1, z_n): Joint posterior distribution of two successive
         latent states

    Args:
        x_list (List[np.ndarray]): List of sequences of observed measurements
        pi (np.ndarray): Current estimated Initial state distribution (K,)
        A (np.ndarray): Current estimated Transition matrix (K, K)
        phi (Dict[np.ndarray]): Current estimated gaussian parameters

    Returns:
        gamma_list (List[np.ndarray]), xi_list (List[np.ndarray])
    """
    n_states = pi.shape[0]
    gamma_list = [np.zeros([len(x), n_states]) for x in x_list]
    xi_list = [np.zeros([len(x)-1, n_states, n_states]) for x in x_list]

    """ YOUR CODE HERE
    Use the forward-backward procedure on each input sequence to populate 
    "gamma_list" and "xi_list" with the correct values.
    Be sure to use the scaling factor for numerical stability.
    """
    K = n_states
    N = len(x_list[0])
    batches = len(x_list)
    alpha = []
    scaling_factor = []

    def __emission(x, phi, K):
        pdf = [scipy.stats.norm.pdf(x, loc=phi['mu'][k], scale=phi['sigma'][k]) for k in range(K)]
        return np.array(pdf).T

    # Initialize alpha
    emission_probs = __emission(x_list, phi, K) # (N * batches * K)
    a =  emission_probs[0] * pi # (batches * K)
    c = np.sum(a, axis=1).reshape(batches, 1) # (batches * 1)
    a /= c
    alpha.append(a)
    scaling_factor.append(c)

    # Forward step
    for n in range(1, N):
        a = alpha[n-1].dot(A) * emission_probs[n]
        c = np.sum(a, axis=1).reshape(batches, 1) # (batches * 1)
        a /= c
        alpha.append(a)
        scaling_factor.append(c)

    alpha = np.array(alpha) # (N * batches * K)

    # Initialize beta
    beta = np.ones((N, batches, K))

    # Backward step
    for n in range(N-2, -1, -1):
        b = (beta[n+1] * emission_probs[n+1]).dot(A.T)
        b /= scaling_factor[n+1]
        beta[n] = b
    
    gamma = alpha * beta
    gamma_list = list(np.transpose(gamma, (1, 0, 2)))

    # Joint Posterior Distribution
    xi = []
    for n in range(1, N):
        a = np.transpose(np.tile(alpha[n-1], (K, 1, 1)).T, (1, 0, 2))
        e = np.transpose(np.tile(emission_probs[n], (K, 1, 1)), (1, 0, 2))
        b = np.transpose(np.tile(beta[n], (K, 1, 1)), (1, 0, 2))
        c = scaling_factor[n]
        x = (a * e * b * A) / c[:, np.newaxis]
        xi.append(x)

    xi = np.array(xi)
    xi_list = list(np.transpose(xi, (1, 0, 2, 3)))

    return gamma_list, xi_list


"""M-step"""
def m_step(x_list, gamma_list, xi_list):
    """M-step of Baum-Welch: Maximises the log complete-data likelihood for
    Gaussian HMMs.
    
    Args:
        x_list (List[np.ndarray]): List of sequences of observed measurements
        gamma_list (List[np.ndarray]): Marginal posterior distribution
        xi_list (List[np.ndarray]): Joint posterior distribution of two
          successive latent states

    Returns:
        pi (np.ndarray): Initial state distribution
        A (np.ndarray): Transition matrix
        phi (Dict[np.ndarray]): Parameters for the Gaussian HMM model, contains
          two fields 'mu', 'sigma' for the mean and standard deviation
          respectively.
    """

    n_states = gamma_list[0].shape[1]
    pi = np.zeros([n_states])
    A = np.zeros([n_states, n_states])
    phi = {'mu': np.zeros(n_states),
           'sigma': np.zeros(n_states)}

    """ YOUR CODE HERE
    Compute the complete-data maximum likelihood estimates for pi, A, phi.
    """
    K = n_states
    N = len(x_list[0])
    batches = len(x_list)
    gamma_list = np.transpose(np.array(gamma_list), (1, 0, 2))
    xi_list = np.array(xi_list)

    # Pi calculation
    pi = gamma_list[0] / np.sum(gamma_list[0], axis=1)[0]
    pi = np.sum(pi, axis=0) / batches

    # A calculation
    sum_e = np.sum(np.sum(xi_list, axis=1), axis=0)
    d = np.sum(sum_e, axis=1)[:, None]
    A = sum_e / d

    # Phi calculation
    d = np.sum(np.sum(gamma_list, axis=0), axis=0)

    x = np.tile(x_list, (K, 1, 1)).T
    mu = np.sum(np.sum(gamma_list * x, axis=0), axis=0) / d
    phi['mu'] = mu

    modified_mu = np.tile(mu, (N, batches, 1))
    sigma = np.sum(np.sum(gamma_list * np.power((x - modified_mu), 2), axis=0), axis=0)
    phi['sigma'] = np.sqrt(sigma / d)

    return pi, A, phi


"""Putting them together"""
def fit_hmm(x_list, n_states):
    """Fit HMM parameters to observed data using Baum-Welch algorithm

    Args:
        x_list (List[np.ndarray]): List of sequences of observed measurements
        n_states (int): Number of latent states to use for the estimation.

    Returns:
        pi (np.ndarray): Initial state distribution
        A (np.ndarray): Time-independent stochastic transition matrix
        phi (Dict[np.ndarray]): Parameters for the Gaussian HMM model, contains
          two fields 'mu', 'sigma' for the mean and standard deviation
          respectively.

    """

    # We randomly initialize pi and A, and use k-means to initialize phi
    # Please do NOT change the initialization function since that will affect
    # grading
    pi, A, phi = initialize(n_states, x_list)

    """ YOUR CODE HERE
     Populate the values of pi, A, phi with the correct values. 
    """
    old = {'pi': pi, 'A': A, 'phi': phi}
    gamma_list, xi_list = e_step(x_list, pi, A, phi)
    pi, A, phi = m_step(x_list, gamma_list, xi_list)
    current = {'pi': pi, 'A': A, 'phi': phi}

    cnt = 0
    while parameter_change(old, current) > 1e-4:
        old = current
        cnt += 1

        gamma_list, xi_list = e_step(x_list, pi, A, phi)
        pi, A, phi = m_step(x_list, gamma_list, xi_list)
        current = {'pi': pi, 'A': A, 'phi': phi}

    print(cnt)

    return pi, A, phi


def parameter_change(old, current):
    return np.max([np.max(np.absolute([
                old['pi'] - current['pi'],
                old['phi']['mu'] - current['phi']['mu'],
                old['phi']['sigma'] - current['phi']['sigma']])),
            np.max(np.absolute(old['A'] - current['A']))])
