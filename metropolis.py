import numpy as np


def metropolis(func, init, Nburnin=0, Nsamp=1, sampleCov=1, args=(), **kwargs):
    """

    Parameters
    ----------
    func: callable
        function to sample from
        this function should take a vector of parameters as first argument
        func() returns a two-element vector, the logPrior and logLike (log base
        10), the sum of which is taken to be the log of the density function
        (i.e. unnormalized posterior).
        (If you don't have this separation, just set func to return one of them
        as zero.)

    init: sequence
        the initial values of which are provided by the real vector thetaInit.

    Nburnin: int
        burning length

    Nsamp: int
        number of final samples after burning period

    sampleCov: sequence
        covariance proposal of the sampling

    Returns
    -------
    lnps: sequence(float)
        lnp values of the samples

    samples: ``ndim, Nsamp``
        samples
    """
    # set up empty sample holder
    ndim = len(init)
    samples = np.zeros((ndim, Nsamp), dtype=float)
    lnps = np.zeros(Nsamp, dtype=float)

    if (np.ndim(sampleCov) == 0) | ((np.ndim(sampleCov) == 1) & (np.size(sampleCov) == ndim)):
        sampleCov = np.eye(ndim) * sampleCov

    # initialize state and log-likelihood
    state = np.array(init).ravel().copy()
    Lp = sum(func(state, *args))

    accepts = 0
    for i in np.arange(0, Nburnin + Nsamp):
        prop = np.random.multivariate_normal(state, sampleCov)
        Lp_prop = sum(func(prop, *args))
        rand = np.random.rand()
        if np.log(rand) < (Lp_prop - Lp):
            accepts += 1
            state = prop.ravel().copy()
            Lp = Lp_prop
        # store only after burning period
        if i >= Nburnin:
            lnps[i - Nburnin] = Lp
            samples[:, i - Nburnin] = state.copy().ravel()

    print('''Metropolis sampler:
    burning: {0:d},
    sampling: {1:d}
    acceptance ratio: {2:0.2f}
    initial guess: {5}
    best lnp: {3:0.4g}
    best params: {4}'''.format(Nburnin, Nsamp,
                               accepts / (Nburnin + Nsamp),
                               lnps.max(),
                               samples[:, lnps.argmax()], init))
    return lnps, samples


def test_metropolis(Nsamp=int(1e4), Nburnin=0, sampleSig=5, thetaInit=-5):
    from scipy import stats
    import pylab as plt

    def testfunc(theta):
        return (stats.cauchy(-10, 2).pdf(theta)
                + 4 * stats.cauchy(10, 4).pdf(theta))

    def lnp(theta):
        """ interface to metrop """
        return (0, np.log10(testfunc(theta)))

    x = np.linspace(-50, 50, 1e4)
    y = testfunc(x)

    # compute normalization function: used later to put on same as histogram
    Zfunc = np.sum(y * x.ptp() / len(x))

    plt.plot(x, y / Zfunc, 'k-', lw=2)
    plt.xlabel('theta')
    plt.ylabel('f')

    lnps, samples = metropolis(lnp, [thetaInit], Nburnin=Nburnin, Nsamp=Nsamp,
                               sampleCov=sampleSig ** 2)

    n, b = np.histogram(samples.ravel(), bins=np.arange(-50, 50, 1),
                        normed=False)
    Zhist = (n * np.diff(b)).sum()
    plt.step(b[1:], n.astype(float) / Zhist, color='r')
    plt.title('test metropolis')
