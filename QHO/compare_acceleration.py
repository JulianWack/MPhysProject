import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
from Oscillator import Oscillator
import time
from datetime import timedelta

plt.style.use('science')
plt.rcParams.update({'font.size': 20})
# plt.rcParams.update({'text.usetex': False}) # faster rendering
mpl.rcParams['axes.prop_cycle'] = cycler(color=['k', 'g', 'b', 'r'])


m = 1
w = 1
N = 1000

# load guesses for eps, ell form varying lattice spacing
a_s = np.logspace(-2,0,30)[::1]
no_accel_eps_guess = np.loadtxt('data/compare_accel/no_accel_paras')[:,2][::1]
no_accel_ell_guess = np.loadtxt('data/compare_accel/no_accel_paras')[:,1][::1]
accel_eps_guess = np.loadtxt('data/compare_accel/accel_paras')[:,2][::1]
accel_ell_guess = np.loadtxt('data/compare_accel/accel_paras')[:,1][::1]
eps_guess = [no_accel_eps_guess, accel_eps_guess]
ell_guess = [no_accel_ell_guess, accel_ell_guess]


accel_bools = [False, True]
no_accel_paras_record = np.empty((len(a_s), 10))
accel_paras_record = np.empty((len(a_s), 10))
records = [no_accel_paras_record, accel_paras_record] # for convenient access to either record array during loop

t1 = time.time()
for i, a in enumerate(a_s):
    for j, accel_bool in enumerate(accel_bools):
        ell = int(ell_guess[accel_bool][i])
        eps = eps_guess[accel_bool][i]
        good_acc_rate = False
        count = 0
        while good_acc_rate == False:
            # print(eps*ell, ell, eps)
            QHO = Oscillator(m, w, N, a, ell, eps)
            QHO.run_HMC(10000, 1, 0.1, accel=accel_bool, store_data=False)
            acc_rate = QHO.acc_rate
            # d_acc_rate = acc_rate - 0.65
            d_acc_rate = 0.65 - acc_rate
            if count >= 10:
                good_acc_rate = True
            # if acceptance rate outwith desired range, change step size proportional to the difference to the optimal acceptance rate of 65%
            if acc_rate < 0.6 or acc_rate > 0.8:
                ell = int(np.rint(ell*(1 + d_acc_rate)))
                eps = 1/ell
                count +=1
                # eps *= 1 + d_acc_rate
                # ell = int(np.ceil(1/eps))
            else:
                good_acc_rate = True

        _, _, _, tau_int_x, tau_int_x_err, dt_x = QHO.correlator(QHO.xs)
        _, _, _, tau_int_x2, tau_int_x2_err, dt_x2 = QHO.correlator(QHO.xs**2)
        print('x: ACF and IAT in, %s'%(str(timedelta(seconds=dt_x))))
        print('x2: ACF and IAT in, %s'%(str(timedelta(seconds=dt_x2))))
        t = QHO.time
        records[j][i] = np.array([a, ell, eps, ell*eps, acc_rate, tau_int_x, tau_int_x_err, tau_int_x2, tau_int_x2_err, t])

    print('-'*32)
    print('Completed %d / %d: a=%.3f'%(i+1, len(a_s), a))
    print('-'*32)

t2 = time.time()
print('Total time: %s'%(str(timedelta(seconds=t2-t1))))
    

# store data 
np.savetxt('data/compare_accel/no_accel_paras', no_accel_paras_record, header='a, ell, eps, ell*eps, acc_rate, tau_int_x, tau_int_x_err, tau_int_x2, tau_int_x2_err, CPU time')
np.savetxt('data/compare_accel/accel_paras', accel_paras_record, header='a, ell, eps, ell*eps, acc_rate, tau_int_x, tau_int_x_err, tau_int_x2, tau_int_x2_err, CPU time')

# load stored results
# no_accel_paras_record = np.loadtxt('data/compare_accel/no_accel_paras')
# accel_paras_record = np.loadtxt('data/compare_accel/accel_paras')

# variables to plot
CPU_time_ratio = accel_paras_record[:,-1]/no_accel_paras_record[:,-1]
no_accel_tau_x, no_accel_tau_x_err = no_accel_paras_record[:,-5], no_accel_paras_record[:,-4]
no_accel_tau_x2, no_accel_tau_x2_err = no_accel_paras_record[:,-3], no_accel_paras_record[:,-2]
accel_tau_x, accel_tau_x_err = accel_paras_record[:,-5], accel_paras_record[:,-4]
accel_tau_x2, accel_tau_x2_err = accel_paras_record[:,-3], accel_paras_record[:,-2]


# fit power law to position autocorrelation time
def lin_func(x, m, b):
    return m*x + b

def fit_log_tau(x, tau, tau_err):
    '''Fit linear function to log of integrated autocorrelation time as function of the variable x which is typically the lattice spacing or the lattice size.

    Returns
    fit: array
        target function evaluated on fitting range with optimal parameter values
    z: float
        the dynamical exponent defined as the negative slope of the fitted function
    z_err: float
        error of the found dynamical exponent 
    '''
    log_tau = np.log(tau)
    log_tau_err = 1/log_tau * tau_err

    popt, pcov = curve_fit(lin_func, x, log_tau, sigma=log_tau_err) # uses chi2 minimization
    z = -popt[0]
    z_err = np.sqrt(pcov[0][0])

    fit = lin_func(x, *popt) 

    return fit, z, z_err


# make CPU time plot
fig = plt.figure(figsize=(8,6))

plt.scatter(a_s, CPU_time_ratio, marker='x')
plt.plot(a_s, CPU_time_ratio)
plt.xscale('log')
plt.xlabel('a')
plt.ylabel(r'$T_{accel}/T_{no\:accel}$')

plt.show()
fig.savefig('plots/compare_accel/CPU_time_N%s.pdf'%N)


# make critical exponent plot for x and x^2
def crit_exp_plot(a_s, no_accel_tau, no_accel_tau_err, accel_tau, accel_tau_err, power):
    '''Plotting routine for showing IAT of position to some power vs lattice spacing and determining the critical exponent

    a_s:    array
        values of the considered lattice spacing
    no_accel_tau, accel_tau:   array
        IAT for HMC and FA HMC respectively for different values of the lattice spacing
    no_accel_tau_err, accel_tau_err:   array
        error of IAT for HMC and FA HMC respectively for different values of the lattice spacing
    power:  int
        power of position for which IAT was computed
    '''
    fig = plt.figure(figsize=(8,6))
    no_accel_fit, no_accel_z, no_accel_z_err = fit_log_tau(a_s, no_accel_tau, no_accel_tau_err) 
    accel_fit, accel_z, accel_z_err = fit_log_tau(a_s, accel_tau, accel_tau_err)

    plt.errorbar(a_s, no_accel_tau, yerr=no_accel_tau_err, fmt='x', capsize=2, color='g', label=r'HMC: $z = %.3f \pm %.3f$'%(no_accel_z, no_accel_z_err))
    plt.plot(a_s, no_accel_tau, color='g')
    plt.errorbar(a_s, accel_tau, yerr=accel_tau_err, fmt='x', capsize=2, color='b', label=r'FA HMC: $z = %.3f \pm %.3f$'%(accel_z, accel_z_err))
    plt.plot(a_s, accel_tau, color='b')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('a')
    if power == 1:
        plt.ylabel(r'$\tau_{int}$ of $x$')
    else:
        plt.ylabel(r'$\tau_{int}$ of $x^%d$'%power)
    plt.legend(prop={'size': 12})

    plt.show()
    fig.savefig('plots/compare_accel/crit_exp_x%s_N%s.pdf'%(power,N))


crit_exp_plot(a_s, no_accel_tau_x, no_accel_tau_x_err, accel_tau_x, accel_tau_x_err, 1)
crit_exp_plot(a_s, no_accel_tau_x2, no_accel_tau_x2_err, accel_tau_x2, accel_tau_x2_err, 2)