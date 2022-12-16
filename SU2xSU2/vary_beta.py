import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
import time
from datetime import timedelta

from SU2xSU2 import SU2xSU2

plt.style.use('science')
plt.rcParams.update({'font.size': 20})
# plt.rcParams.update({'text.usetex': False}) # faster rendering
mpl.rcParams['axes.prop_cycle'] = cycler(color=['k', 'g', 'b', 'r'])


def beta_dependence(betas, save=False):
    '''Perform accelerated and unaccelerated simulations with different values of beta. Optionally can store the simulation results and parameters to load at a later point.
    
    betas: (n,) array
        value of beta to run simulations for
    save: boolean
        specify to store the simulation results
    
    Returns:
    N: int
        lattice size used in simulations
    unaccel_results, accel_results: tuple of (n,8) arrays
        contains: average (and error) action and susceptibility per site as well as their IAT (and error) for the accelerated and unaccelerated simulations respectively
    '''
    # load previous guesses of eps, ell for different beta
    unaccel_eps_guess = np.loadtxt('data/vary_beta/unaccel_record')[:,2]
    unaccel_ell_guess = np.loadtxt('data/vary_beta/unaccel_record')[:,1]
    accel_eps_guess = np.full_like(betas, 0.1) # np.loadtxt('data/vary_beta/accel_record')[:,2]
    accel_ell_guess = np.full_like(betas, 10) # np.loadtxt('data/vary_beta/accel_record')[:,1]
    eps_guess = [unaccel_eps_guess, accel_eps_guess]
    ell_guess = [unaccel_ell_guess, accel_ell_guess]


    accel_bools = [False] # [False, True]
    unaccel_record = np.empty((len(betas), 16))
    accel_record = np.empty((len(betas), 16))
    records = [unaccel_record, accel_record] # for convenient access to either record array during loop

    t1 = time.time()
    for i, beta in enumerate(betas):
        for j, accel_bool in enumerate(accel_bools):
            ell = int(ell_guess[j][i])
            eps = eps_guess[j][i]
            good_acc_rate = False
            count = 0
            while good_acc_rate == False:
                model = SU2xSU2(N=64, a=1, ell=ell, eps=eps, beta=beta)
                model.run_HMC(1000, 20, 0.1, accel=accel_bool, store_data=False)  
                acc_rate = model.acc_rate
                d_acc_rate = 0.65 - acc_rate
                if count >= 10:
                    good_acc_rate = True
                # if acceptance rate outwith desired range, change step size proportional to the difference to the optimal acceptance rate of 65%
                if acc_rate < 0.6 or acc_rate > 0.8:
                    ell = int(np.rint(ell*(1 + d_acc_rate)))
                    eps = 1/ell
                    count +=1
                else:
                    good_acc_rate = True

            s_avg, s_err, IAT_s, IAT_s_err = model.action_per_site() 
            c_avg, c_err, IAT_c, IAT_c_err = model.specific_heat_per_site()
            chi_avg, chi_err, IAT_chi, IAT_chi_err = model.susceptibility_per_site() 
            
            records[j][i] = np.array([beta, ell, eps, acc_rate, s_avg, s_err, IAT_s, IAT_s_err, c_avg, c_err, IAT_c, IAT_c_err, chi_avg, chi_err, IAT_chi, IAT_chi_err])

        print('-'*32)
        print('Completed %d / %d: beta=%.3f'%(i+1, len(betas), beta))
        print('-'*32)

    t2 = time.time()
    print('Total time: %s'%(str(timedelta(seconds=t2-t1))))

    if save:
        np.savetxt('data/vary_beta/unaccel_record', unaccel_record, header='beta, ell, eps, acc_rate, s_avg, s_err, IAT_s, IAT_s_err, c_avg, c_err, IAT_c, IAT_c_err, chi_avg, chi_err, IAT_chi, IAT_chi_err')
        # np.savetxt('data/vary_beta/accel_record', accel_record, header='beta, ell, eps, acc_rate, s_avg, s_err, IAT_s, IAT_s_err, chi_avg, chi_err, IAT_chi, IAT_chi_err')

    return model.N, unaccel_record[:,4:], accel_record[:,4:]
    

def make_plots(N, beta, data, data_label, data_symbol, accel_extension):
    '''make plots showing the beta dependence of the observable passed as data.

    N: int
        lattice size used
    beta: (n,) array
        values of beta for which the observable has been have been computed
    data: (n,4) array
        contains average and error of the observable as well as the associated IAT with error for n simulations at different values of beta.
    data_label: str
        axis label to use for observable
    data_symbol: str
        symbol used in IAT subscript and in file name
    accel_extension: str
        to distinguish figures using accelerated and unaccelerated simulations
    '''
    # observable per site vs beta
    fig = plt.figure(figsize=(8,6))

    plt.errorbar(beta, data[:,0], yerr=data[:,1], fmt='x', capsize=2, color='black')
    plt.plot(beta, data[:,0], color='black')
    plt.xlabel(r'$\beta$')
    plt.ylabel('avg %s per site'%data_label)

    # plt.show()
    fig.savefig('plots/vary_beta/%s_N%d%s.pdf'%(data_symbol.replace('\\', ''), N, accel_extension))

    # observable IAT vs beta
    fig = plt.figure(figsize=(8,6))

    plt.errorbar(beta, data[:,2], yerr=data[:,3], fmt='x', capsize=2, color='black')
    plt.plot(beta, data[:,2], color='black')
    plt.xlabel(r'$\beta$')
    plt.ylabel(r'$\tau_{%s}$'%data_symbol)

    # plt.show()
    fig.savefig('plots/vary_beta/%s_IAT_N%d%s.pdf'%(data_symbol.replace('\\', ''), N, accel_extension))


betas = np.linspace(0.1,5,20)
N, unaccel_res, accel_res = beta_dependence(betas, save=False)

# load stored results (without simulation parameters)
# unaccel_res = np.loadtxt('data/vary_beta/unaccel_record')[:,4:]
# accel_res = np.loadtxt('data/vary_beta/accel_record')[:,4:]

# make plots
make_plots(N, betas, unaccel_res[:,:4], 'action', 's', '')
make_plots(N, betas, unaccel_res[:,4:8], 'specific heat', 'c', '')
make_plots(N, betas, unaccel_res[:,8:], 'susceptibility', '\chi', '')

# make_plots(betas, accel_res[:4], 'action', 's', '_accel')
# make_plots(betas, accel_res[4:], 'susceptibility', '\Chi', '_accel')