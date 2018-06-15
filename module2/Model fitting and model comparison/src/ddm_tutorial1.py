from __future__ import division
import time
import numpy as np
import matplotlib.pyplot as plt


def analytic_ddm_linbound(a1, b1, a2, b2, teval):
    '''
    Calculate the reaction time distribution of a Drift Diffusion model
    with linear boundaries, zero drift, and sigma = 1.
    The upper boundary is y(t) = a1 + b1*t
    The lower boundary is y(t) = a2 + b2*t
    The starting point is 0
    teval is the array of time where the reaction time distribution is evaluated
    Return the reaction time distribution of crossing the upper boundary
    Reference:
    Anderson, Theodore W. "A modification of the sequential probability ratio test
    to reduce the sample size." The Annals of Mathematical Statistics (1960): 165-197.
    Code: Guangyu Robert Yang 2013
    '''

    # Change of variables
    tmp = -2.*((a1-a2)/teval+b1-b2)

    # Initialization
    nMax     = 100  # Maximum looping number
    errbnd   = 1e-7 # Error bound for looping
    suminc   = 0
    checkerr = 0

    for n in range(nMax):
        # increment
        inc = np.exp(tmp*n*((n+1)*a1-n*a2))*((2*n+1)*a1-2*n*a2)-\
              np.exp(tmp*(n+1)*(n*a1-(n+1)*a2))*((2*n+1)*a1-2*(n+1)*a2)
        suminc += inc

        # Break when the relative increment is low for two consecutive updates
        if(max(abs(inc/suminc)) < errbnd):
            checkerr += 1
            if(checkerr == 2):
                break
        else:
            checkerr = 0

    # Probability Distribution of reaction time
    dist = np.exp(-(a1+b1*teval)**2./teval/2)/np.sqrt(2*np.pi)/teval**1.5*suminc;
    dist = dist*(dist>0) # make sure non-negative
    return dist

def analytic_ddm(mu, sigma, b, teval, b_slope=0):
    '''
    Calculate the reaction time distribution of a Drift Diffusion model

    Parameters
    ----------
    mu    : Drift rate
    sigma : Noise intensity
    B     : Constant boundary
    teval : The array of time points where the reaction time distribution is evaluated
    b_slope : (Optional) If provided, then the upper boundary is B(t) = b + b_slope*t,
              and the lower boundary is B(t) = -b - b_slope*t
    Returns
    -------
    dist_cor : Reaction time distribution at teval for correct trials
    dist_err : Reaction time distribution at teval for error trials
    '''
    # Scale B, mu, and (implicitly) sigma so new sigma is 1
    b       /= sigma
    mu      /= sigma
    b_slope /= sigma

    # Get valid time points (before two bounds collapsed)
    teval_valid = teval[b+b_slope*teval>0]

    dist_cor = analytic_ddm_linbound(b, -mu+b_slope, -b, -mu-b_slope, teval_valid)
    dist_err = analytic_ddm_linbound(b,  mu+b_slope, -b,  mu-b_slope, teval_valid)

    # For invalid time points, set the probability to be a very small number
    if len(teval_valid) < len(teval):
        eps = np.ones(len(teval)-len(teval_valid)) * 1e-100
        dist_cor = np.concatenate((dist_cor,eps))
        dist_err = np.concatenate((dist_err,eps))

    return dist_cor, dist_err

def sim_DDM_constant(mu, sigma, B, dt=1, tMax=2500, seed=0):
    """
    Function that simulates one trial of the constant bound DDM

    Parameters
    ----------
    mu: float
        DDM drift rate
    sigma: float
        DDM standard-deviation
    B: float
        DDM boundary
    dt: float, optional
        time step in msec with which DDM will be integrated
    tMax: float, optional
        DDM is integrated from t=0 to t=tMax [in msec], should be multiple of dt
    seed: integer, optional
        random seed

    Returns
    -------
    choice: categorical
        indicates whether left or right boundary was reached by decision variable
    correct: bool
        whether or not the left boundary (which is assumed to be the target boundary) was chosen
    rt: float
        reaction time in msec
    dvTrace: list
        trace of decision variable
    tTrace: array_like
        times at which decision variable was sampled in the simulation

    """

    # Set random seed
    np.random.seed(seed)

    # Additional parameters
    n_max    = tMax / dt   # maximum number of time steps
    tSimu   = dt * np.arange(1,n_max)

    sigma_dt = sigma * np.sqrt(dt)
    mu_dt    = mu * dt

    # Initialize decision variable x
    x = 0

    # Storage
    tTrace = [0]
    dvTrace = [x]

    # Looping through time
    for t in tSimu:
        x += mu_dt + sigma_dt * np.random.randn() # internal decision variable x

        tTrace.append(t)
        dvTrace.append(x) # save new x

        # check boundary conditions
        if x > B:
            rt = t
            choice = 'left'
            break
        if x < -B:
            rt = t
            choice = 'right'
            break
    else: # executed if no break has occurred in the for loop
        # If no boundary is hit before maximum time,
        # choose accoring to decision variable value
        rt = t
        choice = 'left' if x > 0 else 'right'

    correct = (choice == 'left') # suppose left choice is correct

    return choice, correct, rt, dvTrace, tTrace

def plot_rt_distribution(rtCorrect, rtError, bins=None):
    '''
    Plots the reaction time distribution with a bar plot

    Parameters
    ----------
    rtCorrect, rtError : arrays
        reaction times for correct / error trials
    bins: optional
        bins for plotting, can be anything that np.histogram accepts
    '''

    if bins is None:
        maxrt = max((max(rt1),max(rt0)))
        bins = np.linspace(0,maxrt,26)
    countCorrect, bins_edge = np.histogram(rtCorrect, bins=bins)
    countError, bins_edge = np.histogram(rtError, bins=bins)
    n_rt = len(rtCorrect) + len(rtError)

    plt.figure()
    plt.bar(bins_edge[:-1],  countCorrect/n_rt, np.diff(bins_edge), color='blue', edgecolor='white')
    plt.bar(bins_edge[:-1], -countError/n_rt, np.diff(bins_edge), color='red',  edgecolor='white')

    titletxt  = 'Prop. correct {:0.2f}, '.format(sum(countCorrect)/n_rt)
    titletxt += 'Mean RT {:0.0f}/{:0.0f} ms'.format(np.mean(rtCorrect),np.mean(rtError))

    plt.ylabel('Proportion')
    plt.xlabel('Reaction Time')
    plt.title(titletxt)
    plt.xlim((bins.min(),bins.max()))
