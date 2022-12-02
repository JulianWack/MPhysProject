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

        
    def Ham(self, phi, pi):
        '''
        Compute the hamiltonian for a lattice configuration phi, pi
        phi: (N,N,4) array
            parameter values of SU(2) matrices at each lattice site
        pi: (N,N,3) array
            parameter values conjugate momenta at each lattice site
            
        Returns
        H: float
            The Hamiltonian
        '''
        K = 1/2 * np.sum(pi**2) # equivalent to first summing the square of the parameters at each site and then sum over all sites

        phi_hc = SU2.hc(phi)
        phi_NN = phi[self.NN_mask] # (N,N,4,4): containing the 4 paras of each of the 4 NN

        # sum over lattice unit vectors: to the right and up. Hence only need right and top NN, stored at position 0,3 respectively
        G = np.zeros((self.N,self.N))
        for i in [0,3]:
            A = SU2.dot(phi_hc, phi_NN[:,:,i,:])
            G += SU2.tr(A + SU2.hc(A)) # when getting UFuncTypeError, check that dtype of G and SU2.tr is the same (float64 by default)

        # sum over lattice sites    
        S = -1/2 * self.beta * np.sum(G)

        H = K + S

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


    def run_HMC(self, M, thin_freq, burnin_frac, accel=True, store_data=False):
        ''' '''
        np.random.seed(12) # for debugging
        t1 = time.time()
        # Collection of produced lattice configurations. Each one is (N,N,4) such that at each site the 4 parameters describing the associated SU(2) matrix are stored
        configs = np.empty((M+1, self.N, self.N, 4)) 
        delta_Hs = np.empty(M)

        # count accepted samples after burn in, set by start_id
        start_id = int(np.ceil(M*burnin_frac))
        n_acc = 0

        # # cold/ordered start
        # a0 = np.ones((self.N,self.N,1))
        # ai = np.zeros((self.N,self.N,3))
        # configs[0] = np.concatenate([a0,ai], axis=2)

        # # hot start
        # need to assure that norm of parameter vector is 1 to describe SU(2) matrices
        ai = np.random.uniform(-1, 1, size=(self.N,self.N,3))
        a0 = (1 - np.sum(ai**2, axis=2)).reshape((self.N,self.N,1))
        configs[0] = np.concatenate([a0,ai], axis=2)

        with alive_bar(M) as bar:
            for i in range(1, configs.shape[0]):
                phi = configs[i-1]
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
        # print('Finished %d HMC steps in %s'%(M,str(timedelta(seconds=t2-t1))))
        print('Acceptance rate: %.2f%%'%(self.acc_rate*100))
        
        # make final chain of configurations
        # remove starting configuration
        configs = np.delete(configs, 0, axis=0) 
        # Reject burn in and thin remaining chain
        start = start_id+thin_freq-1
        mask = np.s_[start::thin_freq]

        self.configs = configs[mask] # final configurations
        # to plot evolution of an observable, such as delta H, over trajectories store index of trajectories in the final chain
        self.sweeps = np.arange(M)[mask]
        self.delta_Hs = delta_Hs[mask]
        
        if store_data:
            np.save('data/sim_paras.npy', np.array([self.N, self.a, self.ell, self.eps, self.beta]))
            np.save('data/final_chain.npy', self.configs)
            np.save('data/sweeps.npy', self.sweeps)
            np.save('data/dH.npy', self.delta_Hs)
            

    def load_data(self):
        '''Loads in data from previous simulation.
        '''
        self.configs = np.load('data/final_chain.npy')
        self.sweeps = np.load('data/sweeps.npy')
        self.delta_Hs = np.load('data/dH.npy')


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



# model = SU2xSU2(N=16, a=1, ell=7, eps=0.1429, beta=1)
# model.run_HMC(2000, 1, 0.1, s=5)      
# avg , avg_err = model.exp__dH(make_plot=True)
# print(avg, avg_err)