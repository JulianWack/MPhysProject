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
#from astropy.stats import jackknife_stats


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
        '''Derivative of the potential wrt. dimensionless less position.
        '''
        return self.m * self.w**2 * x


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
        U = self.a*np.sum( 0.5*self.m*((np.roll(x,-1)-x)/self.a)**2 + self.V(x) )

        return K + U


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
        p_cur = p_old - 0.5*self.eps/self.a * (self.m*(2*x_old-np.roll(x_old,1)-np.roll(x_old,-1)) + self.a**2*self.dV_dx(x_old))
        x_cur = x_old + self.eps*p_cur 

        # ell-1 alternating full steps
        for n in range(self.ell):
            p_cur = p_cur - self.eps/self.a * (self.m*(2*x_cur-np.roll(x_cur,1)-np.roll(x_cur,-1)) + self.a**2*self.dV_dx(x_cur))
            x_cur = x_cur + self.eps*p_cur
    
        # half step in p
        p_cur = p_cur - 0.5*self.eps/self.a * (self.m*(2*x_cur-np.roll(x_cur,1)-np.roll(x_cur,-1)) + self.a**2*self.dV_dx(x_cur))

        return x_cur, p_cur


    def run_HMC(self, M, thin_freq, burnin_frac, store_data=False):
        '''Perform the HMC algorithm to produce lattice configurations (samples) following the PDF appropriate for the considered potential. 
        Assumes a standard normal distributed momentum.
        The initial configuration is chosen at random and new candidate samples are produced by simulating Hamiltonian dynamics and accepted vai a Metropolis step.
        In order keep the correlation between two configurations minimal, only every thin_freq-th accepted configuration will be used in further computations, leading
        to floor(M/thin_freq) samples. The first M*burnin_frac samples will be declared as burin and thus rejected.
        M: int
            number of HMC iterations and thus total number of generated samples
        thin_freq: int
            defines thinning frequency. Only every thin-freq-th sample produced will be considered henceforth
        burin_frac: float
            fraction of total HMC samples needed for the system to thermalize  
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
        # x_samples[0] = np.zeros(self.N) # cold start
        x_samples[0] = np.random.uniform(-1, 1, size=self.N) # hot start
        
        with alive_bar(M) as bar:
            for i in range(1,x_samples.shape[0]):
                # start iteration with previous sample
                x = x_samples[i-1]
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
        t2 = time.time()
        # print('Finished %d HMC steps in %s'%(M,str(timedelta(seconds=t2-t1))))
        # print('Acceptance rate: %.2f%%'%(self.acc_rate*100)) # ideally close to or greater than 65%
        
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

        fig = plt.figure(figsize=(8,8))
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
            fig = plt.figure(figsize=(10,8))
            plt.plot(self.sweeps, xm_config_avg)
            plt.hlines(0.5, self.sweeps[0], self.sweeps[-1], linestyles='-', color='r')
            plt.xlim(self.sweeps[0],self.sweeps[-1])
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
            fig = plt.figure(figsize=(10,8))
            plt.scatter(self.sweeps, np.exp(-self.delta_Hs), s=2) 
            plt.hlines(1, self.sweeps[0], self.sweeps[-1], linestyles='-', color='r')
            # plt.xlim(0,500)
            plt.xlabel('HMC sweep')
            plt.ylabel('$\exp^{-\delta H}$')
            plt.show()
            # fig.savefig('plots/deltaH_vs_sweeps.pdf')


        # print('<exp(-dH)> = %.5f +/- %.5f '%(exp__dH_avg, exp__dH_avg_err))
        return exp__dH_avg, exp__dH_avg_err


    def gs_energy(self):
        '''Computes ground state energy and the standard error on the mean using the Quantum Virial Theorem and is thus only valid for a large lattice.
        '''
        all_x = self.xs.flatten()
        data = 1/2*all_x*self.dV_dx(all_x) + self.V(all_x)
        self.E0_avg = np.mean(data)
        self.E0_avg_err = np.std(data) / np.sqrt(data.size)
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

        fig, (ax_wavefunc, ax_residual) = plt.subplots(2, 1, figsize=(8,8), sharex=True, gridspec_kw={'height_ratios': [1, 0.5]})

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


    def correlator(self, data, L):
        '''Computes autocorrelation function (in the sense of a statistician) and integrated autocorrelation time for passed data.
        Note that the correlation function is the mean covariance between two elements in the data separated by some fixed number of steps t.
        The computed error is thus the standard error on the mean.
        Correlations between elements in the data which are separated by more than L steps in the data array are assumed to be dominated by noise.
        data: 1D array
            Data to be correlated 
        L: int
            maximal length of considered correlations

        Returns
        ts: array
            array of considered separations between two variables
        autocorr_func: array
            autocorrelation function for spacings 0,1,...L-1
        autocorr_func_err:
            standard error of the mean of the autocorrelation function
        int_autocorr_time: float
            integrated autocorrelation time, showing the number of steps needed in the chain until elements become uncorrelated
        int_autocorr_time_err: float
            standard error on the mean of the autocorrelation time
        delta_t: float
            time needed to compute the autocorrelation function and the integrated autocorrelation time
        '''
        def cov(i,t):
            return np.mean(data[i]*data[i+t]) - np.mean(data[i])*np.mean(data[i+t])
        
        ts = np.arange(0, L) # considered spacings between two elements of the data
        autocov_func = np.full(L, np.nan) 
        autocov_func_err = np.full(L, np.nan) 

        t1 = time.time()
        for t in ts:
            covariances = [cov(i,t) for i in range(0,L-t)]
            if len(covariances) == 0:
                break
            autocov_func[t] = np.mean(covariances)
            autocov_func_err[t] = np.std(covariances) / np.sqrt(len(covariances))

        autocorr_func = autocov_func / autocov_func[0]
        autocorr_func_err = autocov_func_err / autocov_func[0]
      
        int_autocorr_time = 0.5
        break_idx = -1
        for t in ts:
            if t >= np.ceil(4*int_autocorr_time+1):
                break_idx = t
                break
            int_autocorr_time += autocorr_func[t]

        int_autocorr_time_err = np.sum(autocov_func_err[:break_idx])
        t2 = time.time()
        delta_t = t2-t1

        return ts, autocorr_func, autocorr_func_err, int_autocorr_time, int_autocorr_time_err, delta_t

    
    def autocorrelation(self, make_plot=False):
        '''Computes the autocorrelation function between position variables of two configurations in the chain.
        Optionally plots the autocorrelation function. 

        Returns
        int_autocorr_time: float
            integrated autocorrelation time. For a correctly thinned chain should less than 1.
        int_autocorr_time_err: float
            standard error on the mean for the integrated autocorrelation time  
        '''    
        # note that the value of L passed to self.correlator must be smaller than len(self.sweeps)
        ts, autocorr_func, autocorr_func_err, int_autocorr_time, int_autocorr_time_err, delta_t = self.correlator(self.xs, 100)
        # print('Configuration correlation function computed in %s'%(str(timedelta(seconds=delta_t))))

        if make_plot:
            fig = plt.figure(figsize=(12,8))
            plt.errorbar(ts, autocorr_func, yerr=autocorr_func_err, fmt='x', capsize=2)
            plt.yscale('log') # negative vals will not be shown
            plt.xlim(0,200)
            fig.gca().xaxis.set_major_locator(MaxNLocator(integer=True)) # set major ticks at integer positions only
            plt.xlabel('computer time') # configuration separation in chain
            plt.ylabel('autocorrelation function')
            plt.show()
            # fig.savefig('plots/autocorrelation.pdf')

        return int_autocorr_time, int_autocorr_time_err


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
        ts, corr_func, corr_func_err, int_autocorr_time, int_autocorr_time_err, delta_t = self.correlator(self.xs.T, self.N)
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
            fig = plt.figure(figsize=(12,8))
            plt.errorbar(ts, corr_func, yerr=corr_func_err, fmt='x', capsize=2)
            plt.plot(ts[:cut], np.exp(lin_func(ts[:cut], *popt)), label='linear fit')
            fig.gca().xaxis.set_major_locator(MaxNLocator(integer=True)) # set major ticks at integer positions only
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

        E1_E0 = -1/self.a * np.log( (R**2-R**self.N) / (R-R**(self.N-1)) )
        return E1_E0