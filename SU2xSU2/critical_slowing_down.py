import os
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler

from calibrate_paras import calibrate

plt.style.use('science')
plt.rcParams.update({'font.size': 20})
# plt.rcParams.update({'text.usetex': False}) # faster rendering
mpl.rcParams['axes.prop_cycle'] = cycler(color=['k', 'g', 'b', 'r'])


# analyze critical slowing down
def power_law(x, z, c):
    return c*x**z

def fit_IAT(xi, IAT, IAT_err):
    '''
    Fit power law for integrated autocorrelation time as function of the correlation length xi.

    Returns
    popt: list length 2
        optimal parameters of fitted function
    z: float
        the dynamical exponent of xi
    z_err: float
        error of the found dynamical exponent 
    '''
    popt, pcov = curve_fit(power_law, xi, IAT, sigma=IAT_err, absolute_sigma=True)
    z = popt[0]
    z_err = np.sqrt(pcov[0][0])

    return popt, z, z_err


def chi_IAT_scaling():
    '''
    Computes the IAT of the susceptibility for the beta,N value pairs used in the asymptotic scaling plot.
    The computation is done for accelerated and unaccelerated dynamics to highlight the difference in scaling. The acceleration mass parameter is chosen as
    the inverse of the fitted correlation length, which was found to yield close to optimal acceleration.
    As a lot of data is necessary to obtain reasonably small errors of the integrated autocorrelation time, the chain is broken up into batches reducing the amount of memory required.
    '''
    a = 1
    Ns, betas, xis, _, _ = np.loadtxt('data/corlen_beta.txt')
    # betas = np.array([0.7333, 0.9333, 1.0333, 1.2, 1.3333])
    # Ns = np.array([40, 64, 96, 160, 256])

    burn_in_amount = 1000
    batch_size = 5000 # number of trajectories in each batch apart from first one where burn_in_amount less
    n_batches = 4 # number of batches
    total_M = n_batches*batch_size - burn_in_amount # total number of simulated trajectories

    # store average of all batches for each beta using acceleration or not
    des_str = ['Unaccelerated dynamics with %d total trajectories \nN, beta, IAT and error, chi and error, simulation time'%total_M, 
                'Accelerated dynamics with %d total trajectories \nN, beta, IAT and error, chi and error, simulation time'%total_M]
    accel_bool = [False, True]
    n = len(betas)
    IATs, IATs_err = np.zeros((2,n)), np.zeros((2,n))
    chis, chis_err = np.zeros((2,n)), np.zeros((2,n))
    times = np.zeros((2,n))
    # location of directories for results from individual batches
    data_path = ['data/crit_slowing/unaccel', 'data/crit_slowing/accel']


    for k, accel in enumerate(accel_bool):
        # starting guess for ell, eps suitable for small beta, N
        prev_ell, prev_eps = 7, 1/7
        for i, (beta, N, xi) in enumerate(zip(betas, Ns, xis)):            
            # make directory to store results from all batches for current beta
            beta_str = str(np.round(beta, 4)).replace('.', '_')
            batch_path = data_path[k]+'/beta_%s'%beta_str
            os.makedirs(batch_path, exist_ok=True)

            # calibrate parameters ell and eps and run first batch
            model_paras_guess = {'N':N, 'a':a, 'ell':prev_ell, 'eps':prev_eps, 'beta':beta, 'mass':1/xi}
            paras_calibrated = calibrate(model_paras_guess, accel=accel)
            sim_paras = {'M':batch_size, 'thin_freq':1, 'burnin_frac':burn_in_amount/batch_size, 'accel':accel, 'store_data':False}
            model, model_paras = calibrate(paras_calibrated, sim_paras, production_run=True)
            prev_ell, prev_eps = model_paras['ell'], model_paras['eps'] # passed to run with next value of beta to calibrate more quickly
            sim_paras['burnin_frac'] = 0.0 # no burn in for the following batches

            for j in range(n_batches):
                if j != 0: # since first batch simulated outside of loop
                    # initialise next batch with end configuration and RGN state
                    sim_paras.update({'starting_config':model.configs[-1], 'RGN_state':model.RGN_state})
                    model.run_HMC(**sim_paras) 

                # compute and store measurements for current batch
                IAT, IAT_err, chi, chi_err = model.susceptibility_IAT()
                sim_time = model.time
                file_path = batch_path+'/batch_%d.txt'%(j+1)
                np.savetxt(file_path, [IAT, IAT_err, chi, chi_err, sim_time], header='beta=%s, N=%d: IAT, IAT error, chi, chi error, simulation time for this batch in sec'%(beta_str, N))
                print('beta=%.4f: %d / %d batches completed'%(beta, j+1, n_batches))

            # average results from batches
            data = np.zeros((n_batches,5))
            for j in range(n_batches):
                file_path = batch_path+'/batch_%d.txt'%(j+1)
                data[j] = np.loadtxt(file_path)

            IATs[k,i], chis[k,i] = np.mean(data[:,0]), np.mean(data[:,2])
            IATs_err[k,i], chis_err[k,i] = np.sqrt( np.sum(data[:,1]**2)/n_batches ), np.sqrt( np.sum(data[:,3]**2)/n_batches )
            times[k,i] = np.sum(data[:,4])

            # store results so far obtained at different beta for currently considered case of acceleration  
            np.savetxt(data_path[k]+'.txt', np.row_stack((Ns, betas, IATs[k], IATs_err[k], chis[k], chis_err[k], times[k])), header=des_str[k])
        print('Completed accel=%s'%accel)


    # range to fit
    cut = -2
    # get critical exponent
    fits = np.zeros((2,xis[:cut].shape[0]))
    zs, zs_err = np.zeros(2), np.zeros(2) 
    red_chi2s = np.zeros(2)
    for k in range(2):
        popt, zs[k], zs_err[k] = fit_IAT(xis[:cut], IATs[k][:cut], IATs_err[k][:cut])
        fits[k] = power_law(xis[:cut], *popt)
        r = IATs[k][:cut] - fits[k]
        red_chi2s[k] = np.sum((r/IATs[k][:cut])**2) / (fits[k].size - 2) # dof = number of observations - number of fitted parameters


    fig = plt.figure(figsize=(8,6))

    # move one data serires slightly for better visibility
    # plt.errorbar(xis[:-3], IATs[0][:-3], yerr=IATs_err[0][:-3], c='g', fmt='.', capsize=2, label='HMC $z = %.3f \pm %.3f$\n $\chi^2/DoF = %.3f$'%(zs[0],zs_err[0], red_chi2s[0]))
    # plt.errorbar(xis[-3:]+0.5, IATs[0][-3:], yerr=IATs_err[0][-3:], c='g', fmt='.', capsize=2)
    
    plt.errorbar(xis, IATs[0], yerr=IATs_err[0], c='b', fmt='.', capsize=2, label='FA HMC $z = %.3f \pm %.3f$\n $\chi^2/DoF = %.3f$'%(zs[0],zs_err[0], red_chi2s[0]))
    plt.plot(xis[:cut], fits[0], c='g')

    plt.errorbar(xis, IATs[1], yerr=IATs_err[1], c='b', fmt='.', capsize=2, label='FA HMC $z = %.3f \pm %.3f$\n $\chi^2/DoF = %.3f$'%(zs[1],zs_err[1], red_chi2s[1]))
    plt.plot(xis[:cut], fits[1], c='b')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'correlation length $\xi$ [$a$]')
    plt.ylabel(r'autocorrelation time $\tau_{\chi}$')
    plt.legend(prop={'size': 12}, frameon=True)

    # plt.show()
    fig.savefig('plots/crit_slowing.pdf')

chi_IAT_scaling()