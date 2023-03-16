# Class to solve the quantum harmonic oscillator and computing basic properties using HMC

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib as mpl
from cycler import cycler
import time
from datetime import timedelta
from alive_progress import alive_bar
from scipy.optimize import curve_fit
from astropy.stats import jackknife_stats

import correlations


plt.style.use('science')
plt.rcParams.update({'font.size': 20})
# plt.rcParams.update({'text.usetex': False}) # faster rendering
mpl.rcParams['axes.prop_cycle'] = cycler(color=['k', 'g', 'b', 'r'])


class Oscillator():
    '''Solves the 1D oscillator for a particle of mass m in potential V (only position dependent) with position derivative dV_dx using HMC.
    Natural units are chosen and equation are presented in a dimensionless form such that input parameters are pure numbers.
    Various statistics can be computed and plotted, including moments of the position, the energies of the ground state and first excited state, as well as the ground state wavefunction. 
     
    The position of the particle over time is captured at discrete points in time, which can be viewed as a 1D lattice. The vector of positions 
    at t_i is referred to as the lattice configuration and describes the path of the particle. The lattice is assumed to be periodic with N sites and spacing eps.
    Candidate lattice configurations are proposed by solving Hamilton's equations via the leapfrog method for ell steps of size eps.'''


    def __init__(self, m, w, N, a, ell, eps):
        '''
        m: float
            dimensionless effective mass of particle
        w: float 
            dimensionless frequency of harmonic potential
        N: int
            Number of time discretizations of particle path i.e. lattice sites 
        a: float DIMENSIONFUL
            Spacing of lattice sites
        ell: int
            Number of steps to integrate Hamilton's equation, each of size eps
        eps: float DIMENSIONFUL
            Step size for integrating Hamilton's equations
        '''
        # oscillator parameters
        self.m, self.w  = m, w 
        # lattice parameters
        self.N, self.a = int(N), a
        # leapfrog parameters
        self.ell, self.eps = int(ell), eps


    def V(self, x):
        '''Harmonic potential with dimensionless mass, frequency and position.
        '''
        return 0.5 * self.m * self.w**2 * x**2


    def dV_dx(self, x):
        '''Derivative of the potential wrt. dimensionless position.
        '''
        return self.m * self.w**2 * x


    def der_action(self, x):
        '''Derivative of the action wrt. dimensionless position.
        '''
        return 1/self.a * (self.m*(2*x-np.roll(x,1)-np.roll(x,-1)) + self.a**2*self.dV_dx(x))


    def Ham(self, x, p):
        '''
        Compute the hamiltonian for a lattice configuration x with momenta p under the assumption of a periodic lattice.
        x: array
            lattice configuration
        p: array
            auxillary momenta corresponding to lattice configuration
            
        Returns
        H: float
            The Hamiltonian
        '''
        K = 0.5*np.sum(p**2)
        # x_{i+1} - x_i for periodic lattice can be quickly computed using np.roll
        S = self.a*np.sum( 0.5*self.m*((np.roll(x,-1)-x)/self.a)**2 + self.V(x) )

        return K + S


    def modified_Ham(self, x, p):
        '''Analogous to function self.Ham but computes the modified hamiltonian used to accelerate the dynamics.
        '''
        K = 0.5/self.N* np.sum( np.abs(np.fft.fft(p))**2 *  self.A )
        S = self.a*np.sum( 0.5*self.m*((np.roll(x,-1)-x)/self.a)**2 + self.V(x) )

        return K + S


    def prod_A_pi(self, p_F):
        '''Computes the element wise product of the inverse kernel and the momentum in Fourier space.
        In the literature often written as the element wise product of A and pi.
        '''
        return np.multiply(self.A, p_F)


    def leapfrog(self, x_old, p_old):
        '''
        Returns a new candidate lattice configuration (sample) by evolving the last accepted sample though solving Hamilton's equations via the leapfrog scheme.
        x_old: array
            last accepted sample
        p_old: array
            auxillary momenta corresponding to last accepted sample
            
        Returns
        x_cur: array
            final position of simulating Hamiltonian dynamics
        p_cur: array
            final momentum of simulating Hamiltonian dynamics
        '''
        # half step in p, full step in x
        p_cur = p_old - 0.5*self.eps * self.der_action(x_old)
        x_cur = x_old + self.eps*p_cur 

        # ell-1 alternating full steps
        for n in range(self.ell):
            p_cur = p_cur - self.eps * self.der_action(x_cur)
            x_cur = x_cur + self.eps*p_cur
    
        # half step in p
        p_cur = p_cur - 0.5*self.eps * self.der_action(x_cur)

        return x_cur, p_cur


    def FA_leapfrog(self, x_old, p_old):
        '''Analogous to function self.leapfrog but for the modified hamiltonian for which the position update is most efficiently done
        in Fourier space.
        '''
        # half step in p and get FT, full step for x
        p_cur = p_old - 0.5*self.eps * self.der_action(x_old)
        p_cur_F = np.fft.fft(p_cur)
        x_cur = x_old + self.eps * np.real( np.fft.ifft(self.prod_A_pi(p_cur_F)) )

        # ell-1 alternating full steps
        for n in range(self.ell):
            p_cur = p_cur - self.eps * self.der_action(x_cur)
            p_cur_F = np.fft.fft(p_cur)
            x_cur = x_cur + self.eps * np.real( np.fft.ifft(self.prod_A_pi(p_cur_F)) )
    
        # half step in p
        p_cur = p_cur - 0.5*self.eps * self.der_action(x_cur)

        return x_cur, p_cur


    def p_samples(self):
        '''Returns real space sample of momenta according to the distribution based on the modified kinetic term in the modified hamiltonian.
        A lattice of even size is assumed.

        Returns
        pi: array
            samples of the auxillary momentum in real space
        '''
        # momenta in Fourier space
        pi_F = np.zeros(self.N, dtype=complex)

        PI_std = np.sqrt(self.N /  self.A) 
        PI = np.random.normal(loc=0, scale=PI_std) # returns array that matches shape of PI_std

        # assign special modes for which FT exponential becomes +/-1. To get real pi in real space, the modes must be real themselves.
        N_2 = int(self.N/2)
        pi_F[0] = PI[0]
        pi_F[N_2] = PI[N_2]

        # unpack remaining PI into Fourier space pi
        pi_F[1:N_2] = 1/np.sqrt(2) * (PI[1:N_2] + 1j * PI[N_2+1:][::-1])
        # impose hermitean symmetry
        pi_F[N_2+1:] = np.conj(pi_F[1:N_2][::-1])

        # pi is real by construction
        pi = np.real(np.fft.ifft(pi_F))

        return pi


    def kernel_inv_F(self):
        '''Returns inverse of the action kernel computed in the Fourier space.
        Introducing A becomes useful when dealing with higher dimensions.
        '''
        k = np.arange(0, self.N) # lattice in Fourier space
        A = (self.m/self.a * (4*np.sin(np.pi * k/self.N)**2 + (self.a*self.w)**2) )**(-1)

        return A


    def run_HMC(self, M, thin_freq, burnin_frac, accel=True, store_data=False):
        '''Perform the HMC algorithm to produce lattice configurations (samples) following the distribution defined by the action. 
        Using the boolean argument 'accel', one can choose between using the ordinary Hamiltonian for the system (accel=False) or modifying the kinetic term (accel=True) to accelerate the dynamics.
        In the former case, the auxillary momentum distribution is assumed to be a standard normal, while in the latter case the construction is more involved and implemented in the
        function self.p_samples. 
        The initial configuration is obtained from a hot start (hard-coded but can equally use cold start) and new candidate configurations are produced by simulating Hamiltonian dynamics
        and are accepted or rejected via a Metropolis step.
        In order keep the correlation between two configurations minimal, only every thin_freq-th accepted configuration will be used in further computations, leading
        to floor(M/thin_freq) samples. The first M*burnin_frac samples will be declared as burin and thus rejected.
        M: int
            number of HMC iterations and thus total number of generated samples
        thin_freq: int
            defines thinning frequency. Only every thin-freq-th sample produced will be considered henceforth
        burin_frac: float
            fraction of total HMC samples needed for the system to thermalize  
        accel: bool
            By default True, indicating to use the Fourier acceleration
        store_data: bool
            store simulation parameters and data    
        '''

        if type(thin_freq) is not int:
            raise ValueError('The thinning frequency must be an integer.')

        t1 = time.time()
        # each row is one lattice configuration
        x_samples = np.empty((M+1, self.N))
        # difference of Hamiltonian between next and current sample
        delta_Hs = np.empty(M)
        # count accepted samples after burn in, set by start_id
        start_id = int(np.ceil(M*burnin_frac))
        n_acc = 0

        # initial random lattice configuration to seed HMC
        # set seed for reproducibility 
        # np.random.seed(42)
        # x_samples[0] = np.zeros(self.N) # cold start
        x_samples[0] = np.random.uniform(-1, 1, size=self.N) # hot start
        
        if accel:
            self.A = self.kernel_inv_F()

        with alive_bar(M) as bar:
            for i in range(1,x_samples.shape[0]):
                # start iteration with previous sample
                x = x_samples[i-1]
                if accel:
                    p = self.p_samples()
                    x_new, p_new = self.FA_leapfrog(x, p)
                    delta_Hs[i-1] = self.modified_Ham(x_new,-p_new) - self.modified_Ham(x,p)
                else:
                    p = np.random.standard_normal(self.N)
                    x_new, p_new = self.leapfrog(x, p)
                    delta_Hs[i-1] = self.Ham(x_new,-p_new) - self.Ham(x,p)

                acc_prob = np.min([1, np.exp(-delta_Hs[i-1])])

                if acc_prob > np.random.random():
                    x_samples[i] = x_new
                    if i >= start_id:
                        n_acc += 1 
                else:
                    x_samples[i] = x 
                bar()
    
        self.acc_rate = n_acc/(M-start_id)
        self.time = t1-time.time()
        # print('Finished %d HMC steps in %s'%(M,str(timedelta(seconds=t2-t1))))
        print('Acceptance rate: %.2f%%'%(self.acc_rate*100)) # ideally close to or greater than 65%
        
        # remove random starting configuration
        x_samples = np.delete(x_samples, 0, axis=0) 

        # Reject burn in and thin remaining chain to reduce correlation
        start = start_id+thin_freq-1
        mask = np.s_[start::thin_freq]

        self.sweeps = np.arange(M)[mask] # indices of HMC iterations in final chain
        self.xs = x_samples[mask]
        self.delta_Hs = delta_Hs[mask]

        if store_data:
            np.save('data/sim_paras.npy', np.array([self.m, self.w, self.N, self.a, self.ell, self.eps]))
            np.save('data/sweeps.npy', self.sweeps)
            np.save('data/final_chain.npy', self.xs)
            np.save('data/dH.npy', self.delta_Hs)


    def load_data(self):
        '''Loads in data from previous simulation.
        '''
        self.sweeps = np.load('data/sweeps.npy')
        self.xs = np.load('data/final_chain.npy')
        self.delta_Hs = np.load('data/dH.npy')


    def show_one_configuration(self):
        '''Plot one possible path of the particle: position depending on imaginary time.
         Chooses a random accepted sample.
        '''
        sample = self.xs[10] # arbitrary choice
        ts = np.arange(0, self.N*self.eps, self.eps) # integration_time

        fig = plt.figure(figsize=(6,6))
        plt.scatter(sample, ts, marker='x')
        plt.plot(sample, ts)
        plt.xlabel('x')
        plt.ylabel('imaginary time')
        plt.show()
        fig.savefig('plots/example_configuration.pdf')


    def x_moment(self, m, make_plot=False):
        '''Computes m-th position moment i.e. <x^m> and the standard error of the mean.
        First, find the average over time slices within each configuration. Then perform the ensemble average across all configurations to get the moment.
        Optionally, plot the moment for each configuration to get the development of the statistic vs HMC sweeps i.e. MC time.
        '''
        xm_config_avg = np.mean(self.xs**m, axis=1) # moment in each configuration
        xm_avg = np.mean(xm_config_avg) # ensemble average
        xm_err_naive = np.std(xm_config_avg) / np.sqrt(xm_config_avg.size)

        ts, ACF, ACF_err, IAT, IAT_err, delta_t = correlations.autocorrelator(xm_config_avg)
        xm_avg_err =  xm_err_naive * np.sqrt(IAT)

        if make_plot:
            # m=2 discrete theory prediction 
            dis_theo = self.x2_dis_theo()

            fig = plt.figure(figsize=(16,5))
            plt.plot(self.sweeps, xm_config_avg, zorder=1, label=r'HMC $\langle x^{%d} \rangle = %.3f \pm %.3f$'%(m, xm_avg, xm_avg_err))
            plt.hlines(dis_theo, self.sweeps[0], self.sweeps[-1], linestyles='-', color='r', zorder=2, label=r'lattice $\langle x^2 \rangle = %.3f$'%dis_theo)
            plt.xlabel('computer time')
            plt.ylabel(r'$x^{%d}$'%m)
            plt.legend(prop={'size':12}, frameon=True)
            plt.xlim(left=-20, right=1000)
            plt.show()
            # fig.savefig(f'plots/x%d_vs_sweeps.pdf'%m)

        # print('<x^%d> = %.5f +/- %.5f '%(m, xm_avg, xm_avg_err))
        return xm_avg, xm_avg_err


    def x2_dis_theo(self):
        '''Returns the discrete theory result for the expectation value of position squared.
        '''
        A = self.w * np.sqrt(1 + 1/4*(self.a*self.w)**2)
        R = np.sqrt(1 + (self.a*A)**2) - self.a*A
        return 1/(2*self.m*A) * (1+R**self.N) / (1-R**self.N)


    def exp__dH(self, make_plot=False):
        '''Computes the average of exp(-dH) and the standard error on the mean. Note that dH = H_new - H_old is the difference of the Hamiltonian between 
        two consecutive configurations in the final (thinned and burn in rejected) chain. If the chain has thermalised successfully, the average will be close to 1.
        Optionally, plot exp(-dH) against HMC sweeps i.e. MC time.
        '''
        exp__dH_avg = np.mean(np.exp(-self.delta_Hs))
        exp__dH_avg_err = np.std(np.exp(-self.delta_Hs)) / np.sqrt(exp__dH_avg.size)

        if make_plot:
            fig = plt.figure(figsize=(8,6))
            plt.scatter(self.sweeps, np.exp(-self.delta_Hs), s=2) 
            plt.hlines(1, self.sweeps[0], self.sweeps[-1], linestyles='-', color='r')
            plt.xlabel('HMC sweep')
            plt.ylabel('$\exp^{-\delta H}$')
            plt.show()
            # fig.savefig('plots/deltaH_vs_sweeps.pdf')


        # print('<exp(-dH)> = %.5f +/- %.5f '%(exp__dH_avg, exp__dH_avg_err))
        return exp__dH_avg, exp__dH_avg_err


    def gs_energy(self):
        '''Computes ground state energy and the standard error on the mean using the Quantum Virial Theorem and is thus only valid for a large lattice.
        '''
        # specialized for harmonic oscillator with quadratic potential, allowing to use SEM correction through IAT 
        x2_avg, x2_err = self.x_moment(2)
        E0_avg, E0_err = self.m*self.w**2*x2_avg, self.m*self.w**2*x2_err 
        # more general case for arbitrary potential
        # E0 = 1/2*self.xs*self.dV_dx(self.xs) + self.V(self.xs)
        # E0_config = np.mean(E0, axis=1)
        # E0_avg, bias, E0_err, conf_interval = jackknife_stats(E0_config, np.mean, 0.95)
        
        # print('GS energy = %.5f +/- %.5f '%(self.E0_avg, self.E0_err))
        return E0_avg, E0_err


    def gs_energy_dis_theo(self):
        '''Returns discrete theory result for ground state energy.
        '''
        A = self.w * np.sqrt(1 + 1/4*(self.a*self.w)**2)
        return 2*self.m/self.a**2 * (np.sqrt(1 + (self.a*A)**2) - 1) * self.x2_dis_theo()


    def plot_wavefunction(self, Nbins):
        '''Plots the wave function by producing a histogram over the position of all configurations and all time slices within them.
        Nbins: int
            number of bins used in histogram
        '''

        def discrete_func(x):
            '''analytic wave function from discrete theory'''
            A = self.w * np.sqrt(1 + 1/4*(self.a*self.w)**2)
            return (self.m*A/np.pi)**(0.25) *np.exp(-0.5*self.m*A*x**2)

        def cts_func(x):
            '''analytic wave function from continuous theory'''
            return (self.m*self.w/np.pi)**(0.25) * np.exp(-0.5*self.m*self.w*x**2)

        fig = plt.figure(figsize=(8,6))

        # split data into chunks and make histogram with the same number of bins for each
        # can thus find average and std for each bin hight  
        # randomize position data first to avoid chunks representing parts of the chain (potentially bias). Note that shuffled all_data replaces old all_data
        all_data = self.xs.flatten()
        np.random.shuffle(all_data)
        Nsplits = 100 # more chunks lead to smaller error bars for wavefunction
        data_chunks = np.array_split(all_data, Nsplits)  
        bin_heights = np.full((len(data_chunks), Nbins), np.nan)
        bin_mids = np.full((len(data_chunks), Nbins), np.nan)

        for i,chunk in enumerate(data_chunks):
            bin_heights[i], bin_edges = np.histogram(chunk, bins=Nbins, density=True)
            bin_mids[i] = (bin_edges[1:] + bin_edges[:-1]) / 2

        HMC_vals = np.mean(bin_heights, axis=0)
        bin_mids_avg = np.mean(bin_mids, axis=0)
        xs_for_analytic = np.linspace(bin_mids_avg[0], bin_mids_avg[-1], 200)

        HMC_vals_err = 1/np.sqrt(Nsplits) * np.std(bin_heights, axis=0)
        plt.errorbar(bin_mids_avg, HMC_vals, yerr=HMC_vals_err, zorder=3, fmt='.', capsize=2, label='HMC')

        lattice = np.abs(discrete_func(xs_for_analytic))**2 
        plt.plot(xs_for_analytic, lattice, zorder=2, label='lattice')
        
        cts_vals = np.abs(cts_func(xs_for_analytic))**2
        plt.plot(xs_for_analytic, cts_vals, zorder=1, label='continuum')

        plt.xlabel('x')
        plt.ylabel('$|\psi_0(x)|^2$')
        plt.legend(prop={'size': 12}, frameon=True)
                
        plt.show()
        # fig.savefig('plots/wavefunction.pdf')


    def correlator(self, data, c=4.0, my_upper=None):
        '''Computes autocorrelation function (ACF) (in the sense of a statistician) and integrated autocorrelation time (IAT) for passed data.
        The covariance function (related to the ACF by a normalization) is the mean covariance between two elements in the data separated by some number of steps t.
        After some separation, the correlation between two elements becomes noise dominated which is assumed to be the case once the auto covariance turns negative for the first time
        unless a manual value for the upper bound is passed. This defines the range over which the ACF is computed. 
        The naive standard error is used to estimate the error of the ACF. IAT is found following Caracciolo and Sokal 1986 and the expression for the error is from Madras and Sokal 1988.
        data: 2D array
            each row represents a sample of a random variable whose correlation we seek to find 
        c: int
            optional; parameter to use in Caracciolo's and Sokal's windowing procedure 
        my_upper: int
            optional; manual value of largest separation for which correlations are assumed to be signal dominated  

        Returns
        ts: array
            array of considered separations between two variables
        ACF: array
            autocorrelation function
        ACF_err:
            error of the autocorrelation function
        IAT: float
            integrated autocorrelation time, showing how many rows in the data lie between uncorrelated samples
        IAT_err: float
            error of the autocorrelation time
        delta_t: float
            time needed to compute the ACF and the IAT
        '''
        def cov(i,t):
            return np.mean(data[i]*data[i+t]) - np.mean(data[i])*np.mean(data[i+t])
        
        num = int(data.shape[0]) # number of observations
        ts = np.arange(0, num) # considered separations of between two observations
        if my_upper is None:
            upper = num # largest separation for which correlation is data dominated; estimated below
        else:
            upper = my_upper

        autocov_func = np.zeros(num) 
        autocov_func_err = np.zeros(num) 

        t1 = time.time()
        for t in ts:
            if (my_upper is not None) & (t == my_upper):
                break 

            covariances = np.array([cov(i,t) for i in range(0,num-t)])
            autocov = np.mean(covariances)
            autocov_err = np.std(covariances) / np.sqrt(len(covariances))

            if (my_upper is None) & (autocov < 0):
                upper = t # should be much larger than IAT
                break

            autocov_func[t], autocov_func_err[t] = autocov, autocov_err

        # autocorrelation function and its error on signal dominated range of observation separations
        ACF = (autocov_func / autocov_func[0])[:upper]
        ACF_err = (autocov_func_err / autocov_func[0])[:upper]
        ts = ts[:upper]
      
        # possible autocorrelation times when halting the computation at different separations
        IATs = 2*np.cumsum(ACF) - 1 # IAT defined as 1 + sum starting from separation=1, but cumsum starts with t=0 for which ACF=1
        break_idx = self.auto_window(IATs, c)
        IAT = IATs[break_idx]
        IAT_err = np.sqrt((4*break_idx+2)/data.shape[0]) * IAT

        t2 = time.time()
        delta_t = t2-t1

        return ts, ACF, ACF_err, IAT, IAT_err, delta_t


    def auto_window(self, IATs, c):
        '''Windowing procedure of Caracciolo, Sokal 1986.
        IATs is array of integrated autocorrelation time when terminated at different separations.
        Returns index of deduced IAT in array IATs
        ''' 
        ts = np.arange(len(IATs)) # all possible separation endpoints
        m =  ts < c * IATs # first occurrence where this is false gives IAT 
        if np.any(m):
            return np.argmin(m)
        return len(IATs) - 1


    def autocorr_func_1d(self, x):
        '''Computes the autocorrelation of a 1D array x using FFT and the Wiener Khinchin theorem.
        As FFTs yield circular convolutions and work most efficiently when the number of elements is a power of 2, pad the data with zeros to the next power of 2. 
        '''
        x = np.atleast_1d(x)
        if len(x.shape) != 1:
            raise ValueError("invalid dimensions for 1D autocorrelation function")

        def next_pow_two(n):
            i = 1
            while i < n:
                i = i << 1 # shifts bits to the left by one position i.e. multiplies by 2 
            return i
        
        n = next_pow_two(len(x))

        # Compute the FFT and then (from that) the auto-correlation function
        f = np.fft.fft(x - np.mean(x), n=2 * n)
        acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
        acf /= 4 * n
        # normalize to get autocorrelation rather than autocovariance
        acf /= acf[0]

        return acf

    def correlator_fast(self, data, c=4.0):
        '''A faster alternative to self.correlator using FFTs.
        Based on the implementation in emcee: https://emcee.readthedocs.io/en/stable/tutorials/autocorr/

        data: 2D array
            each row represents a new sample of correlated observations. Hence data here is data.T in self.correlator
            The autocovariance is computed for each row and the final ACF is estimated as the average of those. 
            An alternative would be to average the rows first and estimate the AFC as the autocovariance of that array (Goodman, Weare 2010)

        Returns same quatities as self.correlator
        '''
        data = data.T # to ensure consistency with self.correlator

        ts = np.arange(data.shape[0])
        t1 = time.time()

        # get ACF and its error
        ACFs = np.zeros_like(data)
        for i,row in enumerate(data):
            ACFs[i] = self.autocorr_func_1d(row)
        ACF = np.mean(ACFs, axis=0)
        ACF_err = np.std(ACFs, axis=0) / np.sqrt(data.shape[0])

        # get all possible IAT and apply windowing
        IATs = 2.0 * np.cumsum(ACF) - 1.0
        break_idx = self.auto_window(IATs, c)
        IAT = IATs[break_idx]
        IAT_err = np.sqrt((4*break_idx+2)/data.shape[0]) * IAT

        t2 = time.time()
        delta_t = t2-t1

        return ts, ACF, ACF_err, IAT, IAT_err, delta_t

    
    def autocorrelation(self, make_plot=False):
        '''Computes the autocorrelation function between position variables of two configurations in the chain.
        Optionally plots the autocorrelation function. 

        Returns
        int_autocorr_time: float
            integrated autocorrelation time. 
        int_autocorr_time_err: float
            error estimate for the integrated autocorrelation time  
        '''    
        ts, ACF, ACF_err, IAT, IAT_err, delta_t = self.correlator_fast(self.xs)
        # print('Configuration correlation function computed in %s'%(str(timedelta(seconds=delta_t))))

        if make_plot:
            fig = plt.figure(figsize=(8,6))
            plt.errorbar(ts, ACF, yerr=ACF_err, fmt='x', capsize=2)
            plt.yscale('log') # negative vals will not be shown
            fig.gca().xaxis.set_major_locator(MaxNLocator(integer=True)) # set major ticks at integer positions only
            plt.xlabel('computer time') # configuration separation in chain
            plt.ylabel('autocorrelation function')
            plt.show()
            # fig.savefig('plots/autocorrelation.pdf')

        return IAT, IAT_err


    def correlation(self, upper=None, make_plot=False):
        '''Computes the correlation function between two position variables on the lattice and plots it.
        Uses these results to estimate the energy difference between the ground state and first excited state which will be returned.

        Returns
        delta_E: float, np.NaN
            E_1 - E_0. NaN when energy difference could not be determined due to failed curve fitting
        delta_E_err: float, np.NaN
            error estimate from curve fitting. NaN when energy difference could not be determined due to failed curve fitting
        '''    
        # find 2pt correlation function and normalise
        corr_func, corr_func_err = correlations.correlator_repeats(self.xs.T, self.xs.T) #self.correlator_fast(self.xs.T)
        corr_func, corr_func_err = corr_func/corr_func[0], corr_func_err/corr_func[0] 
        # ts, corr_func, corr_func_err, IAT, IAT_err, delta_t = self.correlator(self.xs.T, my_upper=self.N) # uses navive correlator implementation
        # print('Position correlation function computed in %s'%(str(timedelta(seconds=delta_t))))

        def fit(d,dE):
            return (np.cosh(dE*(d-self.N/2)) - 1) / (np.cosh(dE*self.N/2) - 1)
        
        sep = np.arange(self.N)
        if upper < 2:
            upper = 2
        mask = sep <= upper
        sep_fit = sep[mask]

        popt, pcov = curve_fit(fit, sep_fit, corr_func[mask], sigma=corr_func_err[mask], absolute_sigma=True, bounds=(0,np.inf)) # uses chi2 minimization
        # factor of a needed as separation in units of the lattice spacing while the correlation length is a physical length  
        delta_E = popt[0] / self.a
        delta_E_err = np.sqrt(pcov[0][0]) / self.a

        if make_plot:
            fig = plt.figure(figsize=(8,6))
            plt.errorbar(sep, corr_func, yerr=corr_func_err, zorder=1, fmt='.', capsize=2)
            plt.plot(sep_fit, fit(sep_fit, *popt), zorder=2, label='cosh fit')
            fig.gca().xaxis.set_major_locator(MaxNLocator(integer=True)) # set major ticks at integer positions only
            plt.ylim(bottom=1e-3, top=2)
            plt.yscale('log') # negative vals will not be shown
            plt.xlabel(r'lattice separation $d$ [$a$]')
            plt.ylabel('correlation function $C(d)$')
            plt.legend(prop={'size': 12}, frameon=True)
            plt.show()
            # fig.savefig('plots/correlation.pdf')

        # print('Position tau_int =  %.5f +/- %.5f'%(int_autocorr_time, int_autocorr_time_err))
        return delta_E, delta_E_err


    def delta_E_dis_theo(self):
        '''Computes difference between first two energy levels based on discrete theory.
        '''
        A = self.w * np.sqrt(1 + 1/4*(self.a*self.w)**2)
        R = np.sqrt(1 + (self.a*A)**2) - self.a*A

        j = 1 # for large N, energy difference hardly changes with j

        E1_E0 = -1/self.a * np.log( (R**(j+1)-R**(self.N-j-1)) / (R**j-R**(self.N-j)) )
        return E1_E0