import numpy as np
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


def internal_energy_coupling_exp(betas, save=False):
    '''Find internal energy per site for different values of beta. Produces a plot comparing the simulation result with the weak and strong coupling expansions (w.c. and s.c.).
    Optionally can store the simulation results and parameters to load at a later point.
    
    betas: (n,) array
        value of beta to run simulations for
    save: boolean
        specify to store the simulation results
    '''
    model_paras = {'N':16, 'a':1, 'ell':None, 'eps':None, 'beta':None}
    sim_paras = {'M':5000, 'thin_freq':20, 'burnin_frac':0.1, 'renorm_freq':10000, 'accel':False, 'store_data':False}
    data = np.empty((len(betas), 8))

    # load previous guesses of eps, ell for different beta
    eps_guess = np.loadtxt('data/coupling_expansion')[:,2]
    ell_guess = np.loadtxt('data/coupling_expansion')[:,1]
    # use default guesses when no results are stored
    # eps_guess = np.full_like(betas, 1/7)
    # ell_guess = np.full_like(betas, 7)

    t1 = time.time()
    for i, beta in enumerate(betas):
        model_paras['ell'] = int(ell_guess[i])
        model_paras['eps'] = eps_guess[i]
        model_paras['beta'] = beta
        # model_paras = calibrate(model_paras) # uncomment for preliminary calibration when using default guesses
        model, paras = calibrate(model_paras, sim_paras, production_run=True)
        e_avg, e_err, IAT_e, IAT_e_err = model.internal_energy_density()
        data[i] = np.array([beta, paras['ell'], paras['eps'], model.acc_rate, e_avg, e_err, IAT_e, IAT_e_err])
        print('-'*32)
        print('Completed %d / %d: beta=%.3f'%(i+1, len(betas), beta))
        print('-'*32)

    t2 = time.time()
    print('Total time: %s'%(str(timedelta(seconds=t2-t1))))

    if save:
        meta_str = 'N=%d, a=%.3f, runs=%d, thinning=%d, burn in fraction=%.2f, acceleration=%s \n'%(model_paras['N'], model_paras['a'], sim_paras['M'], sim_paras['thin_freq'], sim_paras['burnin_frac'], sim_paras['accel'])
        np.savetxt('data/coupling_expansion', data, header=meta_str+'beta, ell, eps, acc_rate, e_avg, e_err, IAT_e, IAT_e_err')

    e_avg, e_err = data[:,4], data[:,5] 

    # strong and weak coupling expansions
    b_s = np.linspace(0,1)
    strong = 1/2*b_s + 1/6*b_s**3 + 1/6*b_s**5

    Q1 = 0.0958876
    Q2 = -0.0670
    b_w = np.linspace(0.6, 4)
    weak = 1 - 3/(8*b_w) * (1 + 1/(16*b_w) + (1/64 + 3/16*Q1 + 1/8*Q2)/b_w**2)

    fig = plt.figure(figsize=(8,6))

    plt.errorbar(betas, e_avg, yerr=e_err, fmt='x', capsize=2, label='HMC')
    plt.plot(b_s, strong, label='s.c.')
    plt.plot(b_w, weak, label='w.c.')
    plt.xlabel(r'$\beta$')
    plt.ylabel('internal energy density')
    plt.legend(prop={'size': 12})

    plt.show()
    # fig.savefig('plots/coupling_exp.pdf')


    return 
    

betas = np.linspace(0.01, 4, 20)
internal_energy_coupling_exp(betas, save=True)