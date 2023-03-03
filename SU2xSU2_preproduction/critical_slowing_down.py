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

def linear_func(x, z, b):
    return z*x + b 

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
    log_IAT = np.log(IAT)
    log_IAT_err = IAT_err / IAT
    popt, pcov = curve_fit(linear_func, np.log(xi), log_IAT, sigma=log_IAT_err, absolute_sigma=True)
    z = popt[0]
    z_err = np.sqrt(pcov[0][0])

    return popt, z, z_err


def chi_IAT_scaling():
    '''
     Computes the IAT of the susceptibility for the beta,N value pairs used in the asymptotic scaling plot.
    The computation is done for accelerated and unaccelerated dynamics to highlight the difference in scaling. The acceleration mass parameter is chosen as
    the inverse of the fitted correlation length, which was found to yield close to optimal acceleration.

    In the current design of the simulation, all configurations are stored in an array which becomes 
    very memory intensive for long runs with large lattices. It is therefore recommended to use the associate script in
    the SU2xSU2 directory for production runs rather than this one. 
    '''
    a = 1
    Ns, betas, xis, _, _ = np.loadtxt('data/corlen_beta.txt')
    M = 20000

    n = len(betas)
    accel_bool = [False, True]
    IATs, IATs_err = np.zeros((2,n)), np.zeros((2,n)) # each row gives IAT for the case considered in accel_bool 
    chis, chis_err = np.zeros((2,n)), np.zeros((2,n))
    times = np.zeros((2,n))

    file_path = ['data/crit_slowing/unaccel.txt', 'data/crit_slowing/accel.txt']
    des_str = ['Unaccelerated dynamics with %d total trajectories \nN, beta, IAT and error, chi and error, simulation time'%M, 
                'Accelerated dynamics with %d total trajectories \nN, beta, IAT and error, chi and error, simulation time'%M] 

    for j, accel in enumerate(accel_bool):
        print('Starting acceleration: %s'%accel)
        # starting guess for ell, eps suitable for small beta, N
        prev_ell, prev_eps = 7, 1/7
        for i, (beta, N, xi) in enumerate(zip(betas, Ns, xis)):
            model_paras = {'N':N, 'a':a, 'ell':prev_ell, 'eps':prev_eps, 'beta':beta, 'mass':1/xi}
            paras_calibrated = calibrate(model_paras, accel=accel)
            sim_paras = {'M':M, 'thin_freq':1, 'burnin_frac':0.05, 'accel':accel, 'store_data':False}
            model, model_paras = calibrate(paras_calibrated, sim_paras, production_run=True)
            prev_ell, prev_eps = model_paras['ell'], model_paras['eps'] # passed to run with next value of beta to calibrate more quickly
            times[j,i] = model.time
            IATs[j,i], IATs_err[j,i], chis[j,i], chis_err[j,i] = model.susceptibility_IAT()

            print('Completed %d / %d: beta=%.3f, N=%d'%(i+1, n, beta, N))
            np.savetxt(file_path[j], np.row_stack((betas, Ns, IATs[j], IATs_err[j], chis[j], chis_err[j], times[j])), header=des_str[j])
        print('Completed accel=%s'%accel)


    # get critical exponent
    cut = -2 # range to fit
    fits = np.zeros((2,xis[:cut].shape[0]))
    zs, zs_err = np.zeros(2), np.zeros(2) 
    red_chi2s = np.zeros(2)
    for k in range(2):
        popt, zs[k], zs_err[k] = fit_IAT(xis[:cut], IATs[k][:cut], IATs_err[k][:cut])
        fits[k] = power_law(xis[:cut], popt[0], np.exp(popt[1]))
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