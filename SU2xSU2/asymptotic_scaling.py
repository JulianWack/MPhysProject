import os
import gc
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
from cycler import cycler

from SU2xSU2 import SU2xSU2
from calibrate_paras import calibrate

plt.style.use('science')
plt.rcParams.update({'font.size': 20})
# plt.rcParams.update({'text.usetex': False}) # faster rendering
mpl.rcParams['axes.prop_cycle'] = cycler(color=['k', 'g', 'b', 'r'])


def get_corlength(ds, ww_cor, ww_cor_err, plot_path):
    '''Identical to fitting and plotting part of SU2xSU2.ww_correlation()
    Fits a cosh to the computed wall to wall correlation function data on the first half of the lattice of even size N.
    The array ds contains the wall separations in units of the lattice spacing and its last element is N. 
    '''
    def fit(x,a):
            return (np.cosh((x-N_2)/a) - 1) / (np.cosh(N_2/a) - 1)

    N_2 = int(ds[-1]/2) 
    ds_2 = ds[:N_2+1]
    # exploit symmetry about N/2 to reduce errors (effectively increasing number of data points by factor of 2)
    cor = 1/2 * (ww_cor[:N_2+1] + ww_cor[N_2:][::-1])
    cor_err = np.sqrt(ww_cor_err[:N_2+1]**2 + ww_cor_err[N_2::-1]**2)

    # defining fitting range 
    mask = cor > 0
    popt, pcov = curve_fit(fit, ds_2[mask], cor[mask], sigma=cor_err[mask], absolute_sigma=True)
    cor_length = popt[0] # in units of lattice spacing
    cor_length_err = np.sqrt(pcov[0][0])

    r = cor[mask] - fit(ds_2[mask], *popt)
    reduced_chi2 = np.sum((r/cor_err[mask])**2) / (mask.size - 1) # dof = number of observations - number of fitted parameters

    # make and store plot
    fig = plt.figure(figsize=(8,6))

    plt.errorbar(ds_2, cor, yerr=cor_err, fmt='.', capsize=2)
    ds_fit = np.linspace(0, ds_2[mask][-1], 500)
    plt.plot(ds_fit, fit(ds_fit,*popt), c='g', label='$\\xi = %.3f \pm %.3f$\n $\chi^2/DoF = %.3f$'%(cor_length, cor_length_err, reduced_chi2))
    plt.yscale('log')
    plt.xlabel(r'wall separation [$a$]')
    plt.ylabel('wall-wall correlation')
    plt.legend(prop={'size':12}, loc='upper right') # location to not conflict with error bars
    fig.gca().xaxis.set_major_locator(MaxNLocator(integer=True)) # set major ticks at integer positions only
    fig.savefig('plots/%s'%plot_path)
    plt.close() # for memory purposes

    return cor_length, cor_length_err, reduced_chi2


def mass_lambda():
    '''
    Computes the mass to lambda parameter ratio for a range of beta values at fixed lattice size and spacing.
    To do so, need to compute correlation function and correlation length. Data can be stored and plots saved.
    The produced graph also shows the prediction from the continuum theory, allowing to deduce the value of beta 
    above which the continuum is well approximated by a finite sized lattice.  

    For large beta, a large lattice and many trajectories are necessary to obtain a good estimate of the correlation length.
    Running a single simulation with many trajectories requires a lot of memory to store all configurations necessary to find the correlation function.
    Instead, simulate several shorter chains (batches) where the last configuration of one chain is the initial configuration of the next. By also passing
    the state of the random number generator, combining the batches becomes equivalent to the long chain.  
    '''
    a = 1
    # Ns = [40, 40, 64, 64, 64, 96, 96, 160, 160, 224, 300, 400]
    # betas = np.array([0.6, 0.6667, 0.7333, 0.8, 0.8667, 0.9333, 1.0, 1.0667, 1.1333, 1.2, 1.2667, 1.3333])
    Ns = [512]
    betas = np.array([1.3333])
    
    burn_in_amount = 1000
    batch_size = 5000 # number of trajectories in each batch apart from first one where burn_in_amount less
    n_batches = 3 # number of batches
    total_M = n_batches*batch_size - burn_in_amount # total number of simulated trajectories
    
    xi = np.zeros(betas.shape[0])
    xi_err = np.zeros(betas.shape[0])
    reduced_chi2 = np.zeros(betas.shape[0])

    # starting guess for ell, eps suitable for small beta, N
    prev_ell, prev_eps = 25, 1/25
    for i,beta in enumerate(betas):
        # directory to store results from all batches for current beta
        beta_str = str(np.round(beta, 4)).replace('.', '_')
        os.makedirs('data/corfunc_beta/beta_%s'%beta_str, exist_ok=True)

        # calibrate parameters ell and eps and run first batch
        model_paras = {'N':Ns[i], 'a':a, 'ell':prev_ell, 'eps':prev_eps, 'beta':beta}
        paras_calibrated = calibrate(model_paras, accel=True)
        sim_paras = {'M':batch_size, 'thin_freq':1, 'burnin_frac':burn_in_amount/batch_size, 'accel':True, 'store_data':False}
        model, model_paras = calibrate(paras_calibrated, sim_paras, production_run=True)
        prev_ell, prev_eps = model_paras['ell'], model_paras['eps'] # passed to run with next value of beta to calibrate more quickly
        sim_paras['burnin_frac'] = 0.0 # no burn in for the following batches

        for j in range(n_batches):
            if j != 0: # since first batch simulated outside of loop
                # initialise next batch with end configuration and RGN state
                sim_paras.update({'starting_config':model.configs[-1], 'RGN_state':model.RGN_state})
                model.run_HMC(**sim_paras) 

            # compute and store correlation function for current batch
            file_path = 'corfunc_beta/beta_%s/batch_%d.txt'%(beta_str, j+1) # relative to data folder
            model.ww_correlation(save_data=True, data_path=file_path)
            print('beta=%.4f: %d / %d batches completed'%(beta, j+1, n_batches))
            # clear memory and create new model with identical parameters
            del model
            gc.collect()
            model = SU2xSU2(**model_paras)

        # find average of correlation function from batches
        cor_funcs, cor_funcs_err = np.zeros((n_batches, model.N+1)), np.zeros((n_batches, model.N+1))
        for j in range(n_batches):
            ds, cor_funcs[j,:], cor_funcs_err[j,:] = np.loadtxt('data/corfunc_beta/beta_%s/batch_%d.txt'%(beta_str, j+1))
        ww_cor = np.mean(cor_funcs, axis=0)
        ww_cor_err = np.sqrt( np.sum(cor_funcs_err**2, axis=0) / n_batches )

        # store correlation function data
        meta_str = 'N=%d, a=%.3f, beta=%.3f, number of batches=%d, total number of configurations=%d'%(Ns[i], a, beta, n_batches, total_M)
        np.savetxt('data/corfunc_beta/beta_%s.txt'%beta_str, np.vstack((ds, ww_cor, ww_cor_err)), header=meta_str+'\nRows: separation in units of lattice spacing, correlation function and its error')
        
        # get correlation length and make plot of correlation function with fit
        xi[i], xi_err[i], reduced_chi2[i] = get_corlength(ds, ww_cor, ww_cor_err, plot_path='corfunc_beta/beta_%s.pdf'%beta_str)
        np.savetxt('data/corlen_beta.txt', np.row_stack((Ns, betas, xi, xi_err, reduced_chi2)), header='lattice size, beta, xi, xi_err, chi-square per degree of freedom')


    data = np.loadtxt('data/corlen_beta.txt')
    _, betas, xi, xi_err, _ = data

    mass_lambda = 1/xi * np.exp(2*np.pi*betas) / np.sqrt(2*np.pi*betas)
    mass_lambda_err = mass_lambda / xi * xi_err
    cts_prediction = 32 * np.exp(np.pi/4) / np.sqrt(np.pi*np.e)

    fig = plt.figure(figsize=(8,6))
    plt.errorbar(betas, mass_lambda, yerr=mass_lambda_err, fmt='.', capsize=2)
    plt.hlines(cts_prediction, betas[0], betas[-1], linestyles='--', color='k')
    plt.xlabel(r'$\beta$')
    plt.ylabel(r'$M / \Lambda_{L,2l}$')

    # plt.show()
    # plt.savefig('plots/asym_scaling.pdf')


mass_lambda()