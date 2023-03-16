import os
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler

import time
from datetime import timedelta

from Oscillator import Oscillator
from calibrate_paras import calibrate
import correlations

plt.style.use('science')
plt.rcParams.update({'font.size': 20})
# plt.rcParams.update({'text.usetex': False}) # faster rendering
mpl.rcParams['axes.prop_cycle'] = cycler(color=['k', 'g', 'b', 'r'])


def x_IAT_scaling(n):
    '''
    Computes the IAT of the position for n different values of the lattice spacing between 0.01 and 1 at fixed lattice size and oscillator parameters.
    The computation is done for accelerated and unaccelerated dynamics to highlight the difference in scaling which is quantified by the fitted value of the 
    critical exponent z. 
    '''
    a_s = np.logspace(0, -2, n)
    m, w, N = 1, 1, 1000
    M = 100000 # number of trajectories to simulate for each value of a
  
    accel_bool = [False, True]
    IATs, IATs_err = np.zeros((2,n)), np.zeros((2,n))
    times, acc_rates = np.zeros((2,n)),np.zeros((2,n))

    file_path = ['data/slowdown/unaccel.txt', 'data/slowdown/accel.txt'] # location for final results depending on use of acceleration
    # create needed directories if the dont exist already
    for path in file_path:
        dir_path = os.path.dirname(path)
        os.makedirs(dir_path, exist_ok=True)

    description = 'dynamics with %d total measurements of the position on lattice of size %d\na, IAT and error, simulation time [sec], acceptance rate'%(M,N)
    des_str = ['Unaccelerated '+description, 'Accelerated '+description]

    prev_ell, prev_eps = [2,2], [1/2, 1/2] 
    for i,a in enumerate(a_s):
        for k, accel in enumerate(accel_bool):
            model_paras = {'m':m, 'w':w, 'N':N, 'a':a, 'ell':prev_ell[k], 'eps':prev_eps[k]}
            paras_calibrated = calibrate(model_paras, accel=accel)
            prev_ell[k], prev_eps[k] = paras_calibrated['ell'], paras_calibrated['eps']

            QHO = Oscillator(**paras_calibrated)
            sim_paras = {'M':M, 'thin_freq':1, 'burnin_frac':0.02, 'accel':accel, 'store_data':False}
            QHO.run_HMC(**sim_paras) 
            # large lattice and small spacing can result in unstable leapfrog and sharply oscillating acceptance rate
            # below loop is a quick and dirty attempt to resolve this
            # acc_flag = False # sufficient acceptance rate
            # while not acc_flag:
            #   QHO.run_HMC(**sim_paras) 
            #   acc_flag = QHO.acc_rate > 0.5

            # get ensemble average and IAT of susceptibility
            _, _, _, IATs[k,i], IATs_err[k,i], _ = correlations.autocorrelator_repeats(QHO.xs)
            times[k,i] = QHO.time
            acc_rates[k,i] = QHO.acc_rate

            np.savetxt(file_path[k], np.row_stack((a_s, IATs[k], IATs_err[k], times[k], acc_rates[k])), header=des_str[k])
        print('-'*32)
        print('Completed %d / %d: a=%.3f'%(i+1, n, a))
        print('-'*32)


def critslowing_plot(n):
    '''Makes critical slowing down plot based on n stored measurements of the position autocorrelation time at different values of the lattice spacing.'''
    def power_law(x, z, c):
        return c*x**z

    def linear_func(x, z, b):
        return z*x + b 

    def fit_IAT(a, IAT, IAT_err):
        '''
        Fit power law for integrated autocorrelation time as function of the lattice spacing a.

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
        popt, pcov = curve_fit(linear_func, np.log(a), log_IAT, sigma=log_IAT_err, absolute_sigma=True)
        z = -popt[0]
        z_err = np.sqrt(pcov[0][0])

        return popt, z, z_err


    IATs, IATs_err = np.zeros((2,n)), np.zeros((2,n))
    times, acc_rates = np.zeros((2,n)), np.zeros((2,n))

    a_s, IATs[0], IATs_err[0], times[0], acc_rates[0] = np.loadtxt('data/slowdown/unaccel.txt')
    a_s, IATs[1], IATs_err[1], times[1], acc_rates[1] = np.loadtxt('data/slowdown/accel.txt')


    cut = None # exclusive upper bound of fit

    # get critical exponent
    fits = np.zeros((2,a_s[:cut].shape[0]))
    zs, zs_err, red_chi2s = np.zeros((3,2)) 
    for k in range(2):
        popt, zs[k], zs_err[k] = fit_IAT(a_s[:cut], IATs[k][:cut], IATs_err[k][:cut])
        fits[k] = power_law(a_s[:cut], popt[0], np.exp(popt[1]))
        r = IATs[k][:cut] - fits[k]
        red_chi2s[k] = np.sum((r/IATs[k][:cut])**2) / (fits[k].size - 2) # dof = number of observations - number of fitted parameters


    # critical exponent plot
    fig = plt.figure(figsize=(16,6))

    # move one data serires slightly for better visibility
    # plt.errorbar(xis, IATs[0], yerr=IATs_err[0], c='g', fmt='.', capsize=2, label='HMC $z = %.3f \pm %.3f$\n $\chi^2/DoF = %.3f$'%(zs[0],zs_err[0], red_chi2s[0]))
    # plt.errorbar(xis+0.5, IATs[0], yerr=IATs_err[0], c='g', fmt='.', capsize=2)
    
    plt.errorbar(a_s, IATs[0], yerr=IATs_err[0], c='b', fmt='.', capsize=2, label='HMC $z = %.3f \pm %.3f$\n $\chi^2/DoF = %.3f$'%(zs[0],zs_err[0], red_chi2s[0]))
    plt.plot(a_s[:cut], fits[0], c='b')
    plt.errorbar(a_s, IATs[1], yerr=IATs_err[1], c='r', fmt='.', capsize=2, label='FA HMC $z = %.3f \pm %.3f$\n $\chi^2/DoF = %.3f$'%(zs[1],zs_err[1], red_chi2s[1]))
    plt.plot(a_s[:cut], fits[1], c='r')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'lattice spacing $a$')
    plt.ylabel(r'autocorrelation time $\tau_x$')
    plt.legend(prop={'size': 12}, frameon=True)
    # plt.show()
    fig.savefig('plots/crit_slowdown/slowdown.pdf')

    # simulation time plot
    fig = plt.figure(figsize=(8,6))

    plt.scatter(a_s, times[0]/times[1], marker='x')
    plt.xlabel(r'lattice spacing $a$')
    plt.ylabel(r'simulation time ratio HMC/FA HMC')
    plt.xscale('log')
    # plt.show()
    fig.savefig('plots/crit_slowdown/sim_time.pdf')

    # cost function plot
    cost_funcs = times/acc_rates * np.sqrt(IATs)
    fig = plt.figure(figsize=(8,6))

    plt.scatter(a_s, cost_funcs[0]/cost_funcs[1], marker='x')
    plt.xlabel(r'lattice spacing $a$')
    plt.ylabel(r'cost function ratio HMC/FA HMC')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()



x_IAT_scaling(15)
critslowing_plot(15)