import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler

from calibrate_paras import calibrate

plt.style.use('science')
plt.rcParams.update({'font.size': 20})
# plt.rcParams.update({'text.usetex': False}) # faster rendering
mpl.rcParams['axes.prop_cycle'] = cycler(color=['k', 'g', 'b', 'r'])


def mass_lambda():
    '''
    Computes the mass to lambda parameter ratio for a range of beta values at fixed lattice size and spacing.
    To do so, need to compute correlation function and correlation length. Data can be stored and plots saved.
    The produced graph also shows the prediction from the continuum theory, allowing to deduce the value of beta 
    above which the continuum is well approximated by a finite sized lattice.  
    '''
    a = 1
    Ns = [40, 40, 64, 64, 64, 96, 96, 160, 160, 224, 300, 400]
    betas = np.array([0.6, 0.6667, 0.7333, 0.8, 0.8667, 0.9333, 1.0, 1.0667, 1.1333, 1.2, 1.2667, 1.3333])
    
    xi = np.zeros(betas.shape[0])
    xi_err = np.zeros(betas.shape[0])
    reduced_chi2 = np.zeros(betas.shape[0])
    for i,beta in enumerate(betas):
        model_paras = {'N':Ns[i], 'a':a, 'ell':15, 'eps':1/15, 'beta':beta}
        paras_calibrated = calibrate(model_paras, accel=True)
        sim_paras = {'M':15000, 'thin_freq':1, 'burnin_frac':1/15, 'accel':True, 'store_data':False}
        model, model_paras = calibrate(paras_calibrated, sim_paras, production_run=True)
        beta_str = str(np.round(beta, 4)).replace('.', '_')
        file_path = 'corfunc_beta/beta_%s.txt'%beta_str
        xi[i], xi_err[i], reduced_chi2[i] = model.ww_correlation(save_data=True, data_path=file_path, make_plot=True, show_plot=False, plot_path='corfunc_beta/beta_%s.pdf'%beta_str)

        np.savetxt('data/corfunc_beta/cor_len_vs_beta.txt', np.row_stack((betas, xi, xi_err, reduced_chi2)), header='betas, xi, xi_err, chi-square per degree of freedom')

    data = np.loadtxt('data/corfunc_beta/cor_len_vs_beta.txt')
    betas, xi, xi_err, _ = data

    mass_lambda = 1/xi * np.exp(2*np.pi*betas) / np.sqrt(2*np.pi*betas)
    mass_lambda_err = mass_lambda / xi * xi_err
    cts_prediction = 32 * np.exp(np.pi/4) / np.sqrt(np.pi*np.e)

    fig = plt.figure(figsize=(8,6))
    plt.errorbar(betas, mass_lambda, yerr=mass_lambda_err, fmt='.', capsize=2)
    plt.hlines(cts_prediction, betas[0], betas[-1], linestyles='--', color='k')
    plt.xlabel(r'$\beta$')
    plt.ylabel(r'$M / \Lambda_{L,2l}$')

    # plt.show()
    plt.savefig('plots/asym_scaling.pdf')


mass_lambda()