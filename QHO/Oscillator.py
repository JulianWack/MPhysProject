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
        np.random.seed(42)
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
        xm_avg_err = np.std(xm_config_avg) / np.sqrt(xm_config_avg.size)

        if make_plot:
            fig = plt.figure(figsize=(8,6))
            plt.plot(self.sweeps, xm_config_avg)
            plt.hlines(0.5, self.sweeps[0], self.sweeps[-1], linestyles='-', color='r')
            plt.xlabel('HMC sweep')
            plt.ylabel(r'$\langle x^{%d} \rangle$'%m)
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
        x_config_avg = np.mean(self.xs, axis=1) # avg position in each configuration
        E0_config = 1/2*x_config_avg*self.dV_dx(x_config_avg) + self.V(x_config_avg)
        self.E0_avg = np.mean(E0_config)
        self.E0_avg_err = np.std(E0_config) / np.sqrt(E0_config.size)
        #estimate, bias, stderr, conf_interval = jackknife_stats(data, np.mean, 0.95)
        
        # print('GS energy = %.5f +/- %.5f '%(self.E0_avg, self.E0_avg_err))
        return self.E0_avg, self.E0_avg_err


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

        fig, (ax_wavefunc, ax_residual) = plt.subplots(2, 1, figsize=(8,6), sharex=True, gridspec_kw={'height_ratios': [1, 0.5]})

        # split data into chunks and make histogram with the same number of bins for each
        # can thus find average and std for each bin hight  
        # randomize position data first to avoid chunks representing parts of the chain (potentially bias). Note that shuffled all_data replaces old all_data
        all_data = self.xs.flatten()
        np.random.shuffle(all_data)
        # avg number of data points per bin in each chunk: len(all_data)(Nchunks*Nbins)
        # typically len(all_data) ~ 10^5, Nbins ~ 10^2, so Nchunks ~ 10
        Nchunks = 10 # more chuncks lead to smaller error bars for wavefunction
        data_chunks = np.array_split(all_data, Nchunks)  
        bin_heights = np.full((Nchunks, Nbins), np.nan)
        bin_mids = np.full((Nchunks, Nbins), np.nan)

        for i,chunk in enumerate(data_chunks):
            bin_heights[i], bin_edges = np.histogram(chunk, bins=Nbins, density=True)
            bin_mids[i] = (bin_edges[1:] + bin_edges[:-1]) / 2
        
        HMC_vals = np.mean(bin_heights, axis=0)
        bin_mids_avg = np.mean(bin_mids, axis=0)
        HMC_vals_err = 1/np.sqrt(Nchunks) * np.std(bin_heights, axis=0)
        ax_wavefunc.errorbar(bin_mids_avg, HMC_vals, yerr=HMC_vals_err, capsize=2, label='HMC')


        dis_vals = np.abs(discrete_func(bin_mids_avg))**2 
        dis_line = ax_wavefunc.plot(bin_mids_avg, dis_vals, label='discrete theory')
        ax_residual.errorbar(bin_mids_avg, HMC_vals-dis_vals, yerr=HMC_vals_err, fmt="x", capsize=2, color=dis_line[0].get_color(), label='HMC - dis theory')
        
        cts_vals = np.abs(cts_func(bin_mids_avg))**2
        cts_line = ax_wavefunc.plot(bin_mids_avg, cts_vals, label='continuous theory', linestyle='dashed')
        ax_residual.errorbar(bin_mids_avg, HMC_vals-cts_vals, yerr=HMC_vals_err, fmt="x", capsize=2, color=cts_line[0].get_color(), label='HMC - cts theory')


        ax_wavefunc.set_ylabel('$|\psi(x)|^2$')
        ax_wavefunc.legend(prop={'size': 12})
        
        ax_residual.set_xlabel('x')
        ax_residual.set_ylabel('residual')
        ax_residual.legend(prop={'size': 12})

        fig.tight_layout()
        plt.show()
        # fig.savefig('plots/wavefunction.pdf')


    def correlator(self, data, my_upper=None):
        '''Computes autocorrelation function (ACF) (in the sense of a statistician) and integrated autocorrelation time (IAT) for passed data.
        The covariance function (related to the ACF by a normalization) is the mean covariance between two elements in the data separated by some number of steps t.
        After some separation, the correlation between two elements becomes noise dominated which is assumed to be the case once the auto covariance turns negative for the first time
        unless a manual value for the upper bound is passed. This defines the range over which the ACF is computed. 
        The naive standard error is used to estimate the error of the ACF. IAT and its error are found based on Joseph section 7.1.
        data: 2D array
            each row represents a sample of a random variable whose correlation we seek to find 
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
      
        IATs = 1/2 + np.cumsum(ACF) # possible autocorrelation times when halting the computation at different separations
        # find biggest separation contributing to IAT according to Joseph eq 7.9. argmax picks finds first occurrence when condition yields True
        # when condition never true, argmax returns 0 (array of False, so first occurrence of largest value is at 0). In this case want to consider all possible contribution to IAT_err
        break_idx = np.argmax(ts >= np.ceil(4*IATs+1))
        if break_idx == 0:
            break_idx = upper-1
            
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
            integrated autocorrelation time. For a correctly thinned chain should less than 1.
        int_autocorr_time_err: float
            standard error on the mean for the integrated autocorrelation time  
        '''    
        ts, ACF, ACF_err, IAT, IAT_err, delta_t = self.correlator(self.xs)
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


    def correlation(self, make_plot=False):
        '''Computes the correlation function between two position variables on the lattice and plots it.
        Uses these results to estimate the energy difference between the ground state and first excited state which will be returned.

        Returns
        delta_E: float, np.NaN
            E_1 - E_0. NaN when energy difference could not be determined due to failed curve fitting
        delta_E_err: float, np.NaN
            error estimate from curve fitting. NaN when energy difference could not be determined due to failed curve fitting
        '''    
        # consider all correlations on the lattice 
        ts, corr_func, corr_func_err, IAT, IAT_err, delta_t = self.correlator(self.xs.T, self.N)
        # print('Position correlation function computed in %s'%(str(timedelta(seconds=delta_t))))

        cut = 5 # number of points considered at most for fitting for the exponential decay of corr_func (empirically determined value) 
        mask = corr_func[:cut]>0 # check that correlation is positive on the fitting range
        sep = ts[:cut][mask]
        log_rho = np.log(corr_func[:cut][mask])
        log_rho_err = 1/corr_func[:cut][mask] * corr_func_err[:cut][mask] # error propagation 
        
        def lin_func(x, m, b):
            return m*x+b
         
        if len(mask) == 0:
            print('Unable to compute delta E as the autocorrelation is negative (noise dominated) for small separations.')
            return np.NaN, np.NaN

        popt, pcov = curve_fit(lin_func, sep, log_rho, sigma=log_rho_err, absolute_sigma=True) # uses chi2 minimization
        # get dimensionless energy difference by introducing factor of a (due to computing correlation depending on index separation rather than separation on the lattice)  
        delta_E = -popt[0] / self.a
        delta_E_err = np.sqrt(pcov[0][0]) / self.a

        if make_plot:
            fig = plt.figure(figsize=(8,6))
            plt.errorbar(ts, corr_func, yerr=corr_func_err, fmt='x', capsize=2)
            plt.plot(ts[:cut], np.exp(lin_func(ts[:cut], *popt)), label='linear fit')
            fig.gca().xaxis.set_major_locator(MaxNLocator(integer=True)) # set major ticks at integer positions only
            plt.xlim(0,18)
            plt.yscale('log') # negative vals will not be shown
            plt.xlabel(r'lattice separation [$a$]')
            plt.ylabel('correlation function')
            plt.legend(prop={'size': 12})
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