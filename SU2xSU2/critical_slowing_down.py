import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
import time
from datetime import timedelta

from SU2xSU2 import SU2xSU2
from calibrate_paras import calibrate

plt.style.use('science')
plt.rcParams.update({'font.size': 20})
# plt.rcParams.update({'text.usetex': False}) # faster rendering
mpl.rcParams['axes.prop_cycle'] = cycler(color=['k', 'g', 'b', 'r'])


# analyze critical slowing down
def lin_func(x, m, b):
    return m*x + b


def fit_log_IAT(betas, IAT, IAT_err):
    '''
    Fit linear function to log of integrated autocorrelation time as function of beta.

    Returns
    popt: list length 2
        optimal parameters of fitted function
    z: float
        the dynamical exponent of beta
    z_err: float
        error of the found dynamical exponent 
    '''
    log_IAT = np.log(IAT)
    log_IAT_err = 1/log_IAT * IAT_err

    popt, pcov = curve_fit(lin_func, betas, log_IAT, sigma=log_IAT_err, absolute_sigma=True)
    z = popt[0]
    z_err = np.sqrt(pcov[0][0])

    return popt, z, z_err


def chi_IAT_scaling():
    '''
    Computes the IAT of the susceptibility for a range of beta values and on sufficiently large lattices to not be subject to finite size effects.
    Through the beta function, changing beta is equivalent to changing the lattice spacing a.
    The computation is done for accelerated and unaccelerated dynamics to highlight the difference in scaling.
    '''
    a = 1
    # betas = np.array([0.6, 0.6667, 0.7333, 0.8, 0.8667, 0.9333, 0.9667, 1.0, 1.0333, 1.0667, 1.1333, 1.2, 1.2667, 1.3333, 1.3667, 1.4, 1.4333])
    # Ns = np.array([32, 32, 40, 40, 64, 64, 64, 64, 96, 96, 128, 160, 192, 224, 512, 512, 512]) 
    betas = np.array([0.7333, 0.9333, 1.0333, 1.2, 1.3333])
    Ns = np.array([40, 64, 96, 160, 256])

    n = len(betas)
    accel_bool = [False, True]
    IATs, IATs_err = np.zeros((2,n)), np.zeros((2,n)) # each row gives IAT for the case considered in accel_bool 
    chis, chis_err = np.zeros((2,n)), np.zeros((2,n))
    times = np.zeros((2,n))
    des_str = 'beta, N, IAT and error, chi and error, simulation time' 

    for i, (beta, N) in enumerate(zip(betas, Ns)):
        for j, accel in enumerate(accel_bool):
            model_paras = {'N':N, 'a':a, 'ell':7, 'eps':1/7, 'beta':beta, 'mass':1/20}
            paras_calibrated = calibrate(model_paras, accel=accel)
            sim_paras = {'M':20000, 'thin_freq':1, 'burnin_frac':0.05, 'accel':accel, 'store_data':False}
            model, model_paras = calibrate(paras_calibrated, sim_paras, production_run=True)
            times[j,i] = model.time
            IATs[j,i], IATs_err[j,i], chis[j,i], chis_err[j,i] = model.susceptibility_IAT()

        print('Completed %d / %d: beta=%.3f, N=%d'%(i+1, n, beta, N))
        np.savetxt('data/crit_slowing/unaccel.txt', np.row_stack((betas, Ns, IATs[0], IATs_err[0], chis[0], chis_err[0], times[0])), header='Unaccelerated dynamics: '+des_str)
        np.savetxt('data/crit_slowing/accel.txt', np.row_stack((betas, Ns, IATs[1], IATs_err[1], chis[1], chis_err[1], times[1])), header='Accelerated dynamics: '+des_str)


    # get critical exponent
    bs = np.linspace(betas[0], betas[-1], 50)
    fits = np.zeros((2,bs.shape[0]))
    zs, zs_err = np.zeros(2), np.zeros(2) 
    for i in range(2):
        popt, zs[i], zs_err[i] = fit_log_IAT(betas, IATs[i], IATs_err[i])
        fits[i] = np.exp(lin_func(bs, *popt))

    fig = plt.figure(figsize=(8,6))

    plt.errorbar(betas, IATs[0], yerr=IATs_err[0], c='g', fmt='.', capsize=2, label='HMC $z = %.3f \pm %.3f$'%(zs[0],zs_err[0]))
    plt.plot(bs, fits[0], c='g')
    plt.errorbar(betas, IATs[1], yerr=IATs_err[1], c='b', fmt='.', capsize=2, label='FA HMC $z = %.3f \pm %.3f$'%(zs[1],zs_err[1]))
    plt.plot(bs, fits[1], c='b')
    plt.yscale('log')
    plt.xlabel(r'$\beta$')
    plt.ylabel(r'$\tau_{\chi}$')
    plt.legend(prop={'size': 12})

    # plt.show()
    fig.savefig('plots/crit_slowing.pdf')

chi_IAT_scaling()