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

import SU2_mat_routines as SU2
import correlations


plt.style.use('science')
plt.rcParams.update({'font.size': 20})
# plt.rcParams.update({'text.usetex': False}) # faster rendering
mpl.rcParams['axes.prop_cycle'] = cycler(color=['k', 'g', 'b', 'r'])


class SU2xSU2():

    def __init__(self, N, a, ell, eps, beta): 
        '''
        N: int
            Number of lattice sites along one dimension 
        a: float
            Lattice spacing
        ell: int
            Number of steps to integrate Hamilton's equation, each of size eps
        eps: float
            Step size for integrating Hamilton's equations
        beta: float
            parameter to control temperature: beta = 1/k_b T
        '''

        # lattice parameters
        self.N, self.a = int(N), a
        # leapfrog parameters
        self.ell, self.eps = int(ell), eps
        # model parameters
        self.beta = beta

        # find mask to index phi giving the parameters of the right, left, top, and bottom nearest neighbor
        self.NN_mask = self.make_NN_mask() 


    def make_NN_mask(self):
        '''Makes mask to apply to phi or pi which then gives the matrix parameter values of the nearest neighbors (NN) for each lattice site.
        Hence phi[self.NN_mask] is of shape (N,N,#neighbors,#parameters) i.e (N,N,4,4).
        
        Returns:
        NN_mask: tuple
            tuple of two (N,N,4,1) arrays, each giving the row and column coordinate for all nearest neighbors
        '''
     
        # make a (N,N,2) array storing the row and col indices of each lattice sites
        grid = np.indices((self.N,self.N)) 
        lattice_coords = grid.transpose(1,2,0)

        # shift lattice coordinates by 1 such that the coordinates at (i,j) are those of the right, left, top, and bottom neighbor of lattice site (i,j)
        # rolling axis=1 by -1 means all columns are moved one step to the left with periodic bcs. Hence value of resulting array at (i,j) is (i,j+1), i.e the coordinates of the right neighbor.
        # all of shape (N,N,2)
        right_n = np.roll(lattice_coords, -1, axis=1)
        left_n = np.roll(lattice_coords, 1, axis=1)
        top_n = np.roll(lattice_coords, 1, axis=0)
        bottom_n = np.roll(lattice_coords, -1, axis=0)

        # for each lattice site, for each neighbor, store row and column coordinates
        # order of neighbors: right, left, top, bottom
        NN = np.empty((self.N,self.N,4,2), dtype=int)
        NN[:,:,0,:] = right_n # row and col indices of right neighbors
        NN[:,:,1,:] = left_n
        NN[:,:,2,:] = top_n
        NN[:,:,3,:] = bottom_n

        # make mask to index phi
        # separate the row and column neighbor coordinates for each lattice site: (N,N,4,1)
        NN_rows = NN[:,:,:,0]
        NN_cols = NN[:,:,:,1]
        NN_mask = (NN_rows, NN_cols)

        return NN_mask 

        
    def action(self, phi):
        '''
        Computes the action for lattice configuration phi
        phi: (N,N,4) array
            parameter values of SU(2) matrices at each lattice site

        Returns
        S: float
            the action
        '''
        phi_hc = SU2.hc(phi)
        phi_NN = phi[self.NN_mask] # (N,N,4,4): containing the 4 paras of each of the 4 NN

        # sum over lattice unit vectors: to the right and up. Hence only need right and top NN, stored at position 0,3 respectively
        G = np.zeros((self.N,self.N))
        for i in [0,3]:
            A = SU2.dot(phi_hc, phi_NN[:,:,i,:])
            G += SU2.tr(A + SU2.hc(A)) # when getting UFuncTypeError, check that dtype of G and SU2.tr is the same (float64 by default)

        # sum over lattice sites    
        S = -1/2 * self.beta * np.sum(G)

        return S


    def susceptibility(self, phi):
        '''
        Computes the susceptibility for lattice configuration phi
        phi: (N,N,4) array
            parameter values of SU(2) matrices at each lattice site

        Returns
        Chi: float
            the susceptibility
        '''
        # find product of phi with phi at every other lattice position y
        G = np.zeros((self.N,self.N))
        for i in range(self.N**2):
            phi_flat = phi.flatten()
            shifted = np.roll(phi_flat, -i)
            phi_at_y = shifted.reshape(phi.shape)
             
            A = SU2.dot(phi, SU2.hc(phi_at_y))
            G += SU2.tr(A + SU2.hc(A))

        Chi = 1/2 * np.sum(G)

        return Chi


    def Ham(self, phi, pi):
        '''
        Computes the Hamiltonian for a lattice configuration phi, pi
        phi: (N,N,4) array
            parameter values of SU(2) matrices at each lattice site
        pi: (N,N,3) array
            parameter values conjugate momenta at each lattice site
            
        Returns
        H: float
            the Hamiltonian as the sum of the action and a kinetic term, quadratic in pi
        '''
        T = 1/2 * np.sum(pi**2) # equivalent to first summing the square of the parameters at each site and then sum over all sites
        S = self.action(phi)
        H = T + S

        return H 


    def pi_dot(self, phi):
        '''Time derivative of pi which is given as i times the derivative of the action wrt. phi.
        pi and pi dot are linear combinations of the Pauli matrices and hence described by 3 real parameters alpha
        '''
        phi_hc = SU2.hc(phi)
        phi_NN = phi[self.NN_mask]
        # need sum of NN pairs along the two lattice unit vectors i.e. right+left and top+bottom
        alpha = np.zeros((self.N, self.N, 3))
        for pos, neg in zip([0,1], [2,3]):
            # sum is proportional to SU2 matrix, allowing to apply the SU2 product routine once proportionality constant has been identified
            sum_in_SU2, prop_const = SU2.sum(phi_NN[:,:,pos,:], phi_NN[:,:,neg,:]) # both are potentially complex but their product is always real
            V = (prop_const * SU2.dot(sum_in_SU2, phi_hc)).real
            alpha += 2*V[:,:,1:] # 3 parameters describing matrix -i(V - V^dagger) for the currently considered direction in the lattice

        return self.beta * alpha


    def exp_update(self, pi_dot_dt):
        '''The update matrix for the field phi is the exponential of a linear combination of generators i.e. an SU(2) element itself.
        SU(2) is special as this exponential can be evaluated exactly.

        Returns:
        update: (N,N,4) array
            parameter vectors of the matrices to update phi
        '''
        return SU2.alpha_to_a(pi_dot_dt)


    def leapfrog(self, phi_old, pi_old):
        '''
        Returns a new candidate lattice configuration and conjugate momenta by evolving the passed configuration and momenta via Hamilton's equations through the leapfrog scheme.
        phi_old: (N,N,4) array
            last accepted sample of SU(2) matrices (specifically their parameter vectors) at each lattice site
        pi_old: (N,N,3) array
            conjugate momenta (specifically their parameter vectors) corresponding to phi_old
            
        Returns:
        phi_cur: (N,N,4) array
            SU(2) matrix parameter vectors after simulating dynamics
        pi_cur: (N,N,3) array
            momenta parameter vectors after simulating dynamics
        '''
        # half step in pi, full step in phi
        pi_dot_dt_half = 0.5*self.eps * self.pi_dot(phi_old)
        pi_cur = pi_old + pi_dot_dt_half
        phi_cur = SU2.dot(self.exp_update(pi_cur*self.eps), phi_old)

        # ell-1 alternating full steps
        for n in range(self.ell):
            pi_dot_dt = self.eps * self.pi_dot(phi_cur)
            pi_cur = pi_cur + pi_dot_dt
            phi_cur = SU2.dot(self.exp_update(pi_cur*self.eps), phi_cur)
    
        # half step in pi
        pi_dot_dt_half = 0.5*self.eps * self.pi_dot(phi_cur)
        pi_cur = pi_cur + pi_dot_dt_half

        return phi_cur, pi_cur


    def run_HMC(self, M, thin_freq, burnin_frac, renorm_freq=10000, accel=True, store_data=False):
        '''Perform the HMC algorithm to generate lattice configurations using ordinary or accelerated dynamics (accel=True).
        A total of M trajectories will be simulated. The final chain of configurations will reject the first M*burnin_frac samples as burn in and 
        only consider every thin_freq-th accepted configuration to reduce the autocorrelation. 
        Due to accumulating rounding errors, unitarity will be broken after some number of trajectories. To project back to the group manifold, all matrices are renormalised 
        every renorm_freq-th trajectory. 
        M: int
            number of HMC trajectories and thus total number of generated samples
        thin_freq: int
            frequency by which chain of generated samples will be thinned
        burin_frac: float
            fraction of total HMC samples needed for the system to thermalize  
        renorm_freq: int
            after how many trajectories are all matrices renormalized. Set to None to never renormalize
        accel: bool
            By default True, indicating to use Fourier Acceleration
        store_data: bool
            store simulation parameters and data    
        '''
        np.random.seed(42) # for debugging
        t1 = time.time()
        # Collection of produced lattice configurations. Each one is (N,N,4) such that at each site the 4 parameters describing the associated SU(2) matrix are stored
        configs = np.empty((M+1, self.N, self.N, 4)) 
        delta_Hs = np.empty(M)

        # count accepted samples after burn in, set by start_id
        start_id = int(np.floor(M*burnin_frac))
        n_acc = 0

        # # cold/ordered start
        # a0 = np.ones((self.N,self.N,1))
        # ai = np.zeros((self.N,self.N,3))
        # configs[0] = np.concatenate([a0,ai], axis=2)

        # # hot start
        # sampling 4 points form 4D unit sphere assures that norm of parameter vector is 1 to describe SU(2) matrices. Use a spherically symmetric distribution such as gaussian
        a = np.random.standard_normal((self.N,self.N,4))
        configs[0] = SU2.renorm(a)

        with alive_bar(M) as bar:
            for i in range(1, configs.shape[0]):
                phi = configs[i-1]   
                if renorm_freq is not None:
                    if i % renorm_freq == 0:
                        SU2.renorm(phi)

                # the conjugate momenta are linear combination of Pauli matrices and thus described by 3 parameters
                pi = np.random.standard_normal((self.N,self.N,3))

                phi_new, pi_new = self.leapfrog(phi, pi)
                delta_Hs[i-1] = self.Ham(phi_new,-pi_new) - self.Ham(phi,pi)
                acc_prob = np.min([1, np.exp(-delta_Hs[i-1])])

                if acc_prob > np.random.random():
                    configs[i] = phi_new
                    if i >= start_id:
                        n_acc += 1 
                else:
                    configs[i] = phi 
                bar()
    
        self.acc_rate = n_acc/(M-start_id)
        self.time = t1-time.time()
        # print('Finished %d HMC steps in %s'%(M,str(timedelta(seconds=self.time))))
        print('Acceptance rate: %.2f%%'%(self.acc_rate*100))
        
        # make final chain of configurations
        # remove starting configuration
        configs = np.delete(configs, 0, axis=0) 
        # Reject burn in and thin remaining chain
        start = start_id+thin_freq-1
        mask = np.s_[start::thin_freq]

        self.configs = configs[mask] # final configurations (M,N,N,4)
        # to plot evolution of an observable, such as delta H, over trajectories store index of trajectories in the final chain
        self.sweeps = np.arange(M)[mask]
        self.M = self.sweeps.shape[0] # number of observations
        self.delta_Hs = delta_Hs[mask]
        
        if store_data:
            np.savetxt('data/single_run/model_paras.txt', [self.N, self.a, self.ell, self.eps, self.beta], header='N, a, ell, eps, beta')
            np.savetxt('data/single_run/sim_paras.txt', [M, thin_freq, burnin_frac, self.acc_rate], header='M, thin freq, burn in, accept rate')
            np.save('data/single_run/final_chain.npy', self.configs)
            np.save('data/single_run/sweeps.npy', self.sweeps)
            np.save('data/single_run/delta_Hs.npy', self.delta_Hs)
            

    def load_data(self):
        '''Loads in data from previous simulation.
        '''
        self.configs = np.load('data/single_run/final_chain.npy')
        self.delta_Hs = np.load('data/single_run/delta_Hs.npy')
        self.sweeps = np.load('data/single_run/sweeps.npy')
        self.M = self.sweeps.shape[0]


    def exp__dH(self, make_plot=False):
        '''Computes the average of exp(-dH) and the standard error on the mean. Note that dH = H_new - H_old is the difference of the Hamiltonian between 
        two consecutive configurations in the final (thinned and burn in rejected) chain. If the chain has thermalised successfully, the average will be close to 1.
        Optionally, plot exp(-dH) against HMC sweeps i.e. MC time.
        '''
        exp__dH_avg, _, exp__dH_err, _ = jackknife_stats(np.exp(-self.delta_Hs), np.mean, 0.95)

        if make_plot:
            fig = plt.figure(figsize=(8,6))
            plt.scatter(self.sweeps, np.exp(-self.delta_Hs), s=2) 
            plt.hlines(1, self.sweeps[0], self.sweeps[-1], linestyles='-', color='r')
            plt.xlabel('HMC sweep')
            plt.ylabel('$\exp^{-\delta H}$')
            plt.show()
            # fig.savefig('plots/deltaH_vs_sweeps.pdf')

        return exp__dH_avg, exp__dH_err


    def order_parameter(self, make_plot=False):
        ''' Finds the average matrix of each configuration, The normalized parameter vector of this average matrix is used as the (vectorial) order parameter. 
        Alternative choice described in eq 3.1 of https://www.sciencedirect.com/science/article/pii/0550321382900657
        As no phase transition is present in this model, the order parameter should be close to zero.
        Plots evolution over trajectories and a histogram of the component values for all parameter vectors.

        Returns
        ms_avg: (4,) array
            the order parameter when averaged over all configurations 
        ms_err: (4,) array
            error on the average 
        '''
        avg_SU2 = np.mean(self.configs, axis=(1,2)) # (M,4)
        norm = np.sqrt(np.sum(avg_SU2**2, axis=-1)).reshape((avg_SU2.shape[0], 1))
        ms = np.divide(avg_SU2, norm, out=np.zeros_like(avg_SU2), where=norm!=0)

        ms_avg, ms_err = np.empty(4), np.empty(4)
        ms_int = np.empty((4,2))
        for i in range(4):
            # Jackknife error has tendency to be smaller than correction of SEM via IAT
            # ms_avg[i], _, ms_err[i], ms_int[i] = jackknife_stats(ms[:,i], np.mean, 0.95)
            ts, m_ACF, m_ACF_err, m_IAT, m_IAT_err, delta_t = correlations.autocorrelator(ms[:,i])
            ms_avg[i] = np.mean(ms[:,i])
            ms_err[i] = np.sqrt(m_IAT/self.M) * np.std(ms[:,i])

        if make_plot:
            # evolution over trajectories
            fig, axes = plt.subplots(2, 2, figsize=(12,8), sharex='col')
            axs = axes.flatten() 

            cs=['k', 'g', 'b', 'r']
            for i, ax in enumerate(axs):
                ax.scatter(self.sweeps, ms[:,i], s=2, color=cs[i]) 
                ax.hlines(ms_avg[i], self.sweeps[0], self.sweeps[-1], linestyles='--', color=cs[i])
                ax.fill_between(self.sweeps, ms_avg[i]-ms_err[i], ms_avg[i]+ms_err[i], color=cs[i], alpha=0.2)
                # ax.fill_between(self.sweeps, ms_int[i][0], ms_int[i][1], color=cs[i], alpha=0.2)
                if i in [2,3]:
                    ax.set_xlabel('HMC sweep')
                ax.set_ylabel('$m_{%d}$'%i)
        
            fig.tight_layout()
            plt.show()
            # fig.savefig('plots/m_vs_sweeps.pdf')

            # histogram
            fig = plt.figure(figsize=(8,6))
            all_paras = self.configs.reshape((-1,4))
            norm = np.sqrt(np.sum(all_paras**2, axis=-1)).reshape((all_paras.shape[0], 1))
            normed_paras = np.divide(all_paras, norm, out=np.zeros_like(all_paras), where=norm!=0)

            for i in range(4):
                plt.hist(normed_paras[:,i], bins='auto', density='True', histtype='step', label='$m_{%d}$'%i)
            plt.xlabel('component value')
            plt.ylabel('probability density')
            plt.legend(prop={'size': 12})
            plt.show()
            # fig.savefig('plots/m_histogram.pdf')


        return ms_avg, ms_err


    def internal_energy_density(self, get_IAT=True):
        '''Computes the internal energy (action) per site for each accepted lattice configuration and optionally finds the associated IAT.
        Returns:
        e_avg: float
            action per site when averaged over all accepted configurations
        e_err: float
            Jackknife error of average
        IAT: float
            integrated autocorrelation time for action per site
        IAT_err float
            error of IAT 
        '''
        e = np.empty(self.M) # internal energy density for all accepted configurations
        for i,phi in enumerate(self.configs):
            e[i] = self.action(phi) / (-self.beta * 4 * self.N**2) # definition of internal energy in terms of the action

        # e_avg, _, e_err, _ = jackknife_stats(e, np.mean, 0.95)
        ts, ACF, ACF_err, IAT, IAT_err, delta_t = correlations.autocorrelator(e)
        e_avg = np.mean(e)
        e_err = np.sqrt(IAT/self.M) * np.std(e)
        
        if not get_IAT:
            return e_avg, e_err
        
        return e_avg, e_err, IAT, IAT_err


    def specific_heat_per_site(self, get_IAT=True):
        '''Computes the specific heat per site for each accepted lattice configuration and optionally finds the associated IAT.
        The specific heat is deduced as the average covariance of the internal energy following eq 1 of https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.45.1755 
        Returns:
        c_avg: float
            specific heat per site when averaged over all accepted configurations
        c_err: float
            Jackknife error of average
        IAT: float
            integrated autocorrelation time for specific heat per site
        IAT_err float
            error of IAT 
        '''
        c = np.empty(self.M)
        for i,phi in enumerate(self.configs):
            # find internal energy as done in self.action but don't sum over all sites
            phi_hc = SU2.hc(phi)
            phi_NN = phi[self.NN_mask]

            G = np.zeros((self.N,self.N)) # energy at each site
            for k in [0,3]:
                A = SU2.dot(phi_hc, phi_NN[:,:,k,:])
                G += SU2.tr(A + SU2.hc(A)) 

                for j in range(self.N**2):
                    G_flat = G.flatten()
                    shifted = np.roll(G_flat, -j)
                    G_at_y = shifted.reshape(G.shape)

                    c[i] += (np.mean(G*G_at_y) + np.mean(G)**2) / self.N**2 # identical to np.mean(G)*np.mean(G_at_Y)

                c[i] /= self.N**2 # such that we averaged over positions y
    
        # c_avg, _, c_err, _ = jackknife_stats(c, np.mean, 0.95)
        ts, ACF, ACF_err, IAT, IAT_err, delta_t = correlations.autocorrelator(c)
        c_avg = np.mean(c)
        c_err = np.sqrt(IAT/self.M) * np.std(c)

        if not get_IAT:
            return c_avg, c_err

        return c_avg, c_err, IAT, IAT_err        


    def susceptibility_per_site(self, get_IAT=True):
        '''Computes the susceptibility per site for each accepted lattice configuration and optionally finds the associated IAT.
        Returns:
        chi_avg: float
            susceptibility per site when averaged over all accepted configurations
        chi_err: float
            Jackknife error of average
        IAT: float
            integrated autocorrelation time for susceptibility per site
        IAT_err float
            error of IAT 
        '''
        suscept = np.empty(self.M)
        for i,phi in enumerate(self.configs):
            suscept[i] = self.susceptibility(phi) / self.N**2

        # chi_avg, _, chi_err, _ = jackknife_stats(suscept, np.mean, 0.95)
        ts, ACF, ACF_err, IAT, IAT_err, delta_t = correlations.autocorrelator(suscept)
        chi_avg = np.mean(suscept)
        chi_err = np.sqrt(IAT/self.M) * np.std(suscept)

        if not get_IAT:
            return chi_avg, chi_err

        return chi_avg, chi_err, IAT, IAT_err


    def ww_correlation(self, upper, save_data=False, make_plot=False):
        ''' Computes the wall to wall correlation as described in the report via the cross correlation theorem.
        The data can be optionally saved and the correlation function with the fitted exponential plotted.

        upper: int
            up to which separation (exclusive) in units of the lattice spacing the exponential decay will be fitted  
            Due to periodic boundary conditions, the correlation function will be symmetric about N/2. To get a largely uncontaminated exponential decay, should use upper << N/2.
        
        Returns
        cor_length: float
            fitted correlation length in units of the lattice spacing
        cor_length_err: float
            error in the fitted correlation length
        '''
        ds = np.arange(self.N)
        ww_cor = np.zeros(self.N)
        ww_cor_err = np.zeros(self.N)
        t1 = time.time()
        for p in range(self.N):
            for q in range(self.N):
                # effect of transpose added manually: spatial components reversed sign such that correlation from all components add
                for k in range(4):
                    # passes (N,M) arrays: correlations along axis 0 while axis 1 hosts results from repeating the measurement for different configurations
                    cf, cf_err = correlations.correlator_repeats(self.configs[:,p,:,k].T, self.configs[:,q,:,k].T)  
                    # prefactor not necessary as normalization divides it out. Included here to make connection to description in report more apparent
                    ww_cor += 4/self.N**2 * cf
                    ww_cor_err += 4/self.N**2 * cf_err
        # normalize
        ww_cor, ww_cor_err = ww_cor/ww_cor[0], ww_cor_err/ww_cor[0]
        t2 = time.time()
        print('Wall correlation time: %s'%(str(timedelta(seconds=t2-t1))))

        if save_data:
            meta_str = 'N=%d, a=%.3f, beta=%.3f, number of configurations=%d'%(self.N, self.a, self.beta, self.M)
            np.savetxt('data/wallwall_cor.txt', np.vstack((ds, ww_cor, ww_cor_err)), header=meta_str+'\nRows: separation in units of lattice spacing, correlation function and its error')


        def lin_func(x, m, b):
            return m*x+b

        popt, pcov = curve_fit(lin_func, ds[:upper], np.log(ww_cor[:upper]), sigma=np.log(ww_cor_err[:upper]), absolute_sigma=True) # uses chi2 minimization
        cor_length = -1/popt[0] # in units of lattice spacing
        cor_length_err = np.sqrt(pcov[0][0]) # in units of lattice spacing

        if make_plot:
            fig = plt.figure(figsize=(8,6))

            plt.errorbar(ds, ww_cor, yerr=ww_cor_err, fmt='.', capsize=2)
            plt.plot(ds[:upper], np.exp(lin_func(ds[:upper],*popt)), label='$\\xi = %.3f \pm %.3f$'%(cor_length, cor_length_err))
            plt.xlabel(r'lattice separation [$a$]')
            plt.ylabel('wall-wall correlation')
            plt.legend(prop={'size':12})
            fig.gca().xaxis.set_major_locator(MaxNLocator(integer=True)) # set major ticks at integer positions only
            plt.show()
            # fig.savefig('plots/wallwall_correlation.pdf')

        return cor_length, cor_length_err