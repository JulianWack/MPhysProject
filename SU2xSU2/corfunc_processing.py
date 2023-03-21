# routines to process stored correlation function data and make related plots
import numpy as np
from scipy.optimize import curve_fit
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler

from SU2xSU2 import get_avg_error

plt.style.use('science')
plt.rcParams.update({'font.size': 20})
# plt.rcParams.update({'text.usetex': False}) # faster rendering
mpl.rcParams['axes.prop_cycle'] = cycler(color=['k', 'g', 'b', 'r'])


def process_rawchain():
    '''Processes raw correlation function data:
    Finds ensemble average, normalizes data and applies mirroring about N/2 to reduce errors. Resulting data is stored.
    To be used on data from partially compute simulations. 
    '''
    beta_str = '1_4'
    corfunc_chain = np.load('data/corfuncs/rawchains/%s.npy'%beta_str)
    ww_cor, ww_cor_err = get_avg_error(corfunc_chain)
    print('data shape: ', corfunc_chain.shape)

    # normalize and use periodic bcs to get correlation for wall separation of N to equal that of separation 0
    ww_cor, ww_cor_err = ww_cor/ww_cor[0], ww_cor_err/ww_cor[0]
    ww_cor, ww_cor_err = np.concatenate((ww_cor, [ww_cor[0]])), np.concatenate((ww_cor_err, [ww_cor_err[0]]))

    # use symmetry about N/2 due to periodic bcs and mirror the data to reduce errors (effectively increasing number of data points by factor of 2)
    N_2 = int(ww_cor.shape[0]/2) 
    ds = np.arange(N_2+1) # wall separations covering half the lattice length
    cor = 1/2 * (ww_cor[:N_2+1] + ww_cor[N_2:][::-1])
    cor_err = np.sqrt(ww_cor_err[:N_2+1]**2 + ww_cor_err[N_2::-1]**2) / np.sqrt(2)

    np.save('data/corfuncs/beta_%s.npy'%beta_str, np.row_stack([ds, cor, cor_err]))

# process_rawchain()


def effective_mass(beta):
    '''
    Effective mass plot for simulation results at the passed value of beta. This tends to highlight the worst in the data.
    The normalized and mirrored data of the correlation function is loaded to produce the plot.
    The effective mass will be computed based on the assumption that the correlation function follows the shape of
    a cosh (analytically expected due to periodic boundary conditions) or of a pure exponential decay.
    For small separations the two methods should agree, allowing to gauge is the chosen lattice size is too small.
    The cosh assumption will generally produce a noisier plot as each data point considers 3 values of the correlation function while 
    in the decay assumption only two are used.
    '''
    def cosh_corfunc(cor, cor_err):
        '''effective mass and its error based on a cosh correlation function.
        A lattice of even size is assumed.

        cor: (N/2)
            value of wall to wall correlation function on the first half of the lattice
        cor_err: (N/2)
            error of correlation function on the first half of the lattice

        Returns
        m_eff: (N/2,)
            effective mass
        m_eff_err: (N/2)
            error of the effective mass
        '''
        rel_err = cor_err / cor # relative error
        cor_1, cor_err_1 = np.roll(cor, -1), np.roll(cor_err, -1) # shift to d+1
        rel_err_1 = cor_err_1 / cor_1
        cor__1, cor_err__1 = np.roll(cor, 1), np.roll(cor_err, 1) # shift to d-1
        rel_err__1 = cor_err__1 / cor__1

        A, B = cor_1/cor, cor__1/cor
        x = (A+B)/2 
        m_eff = np.arccosh(x)

        delta_x = 1/2 * (A*(rel_err_1 - rel_err) + B*(rel_err__1 - rel_err))
        # delta_x = A/2*(np.sqrt(rel_err_1**2 + rel_err**2)) + B/2*(np.sqrt(rel_err__1**2 + rel_err**2))
        m_eff_err = 1/np.sqrt(x**2-1) * delta_x

        return m_eff, m_eff_err

    def exp_corfunc(cor, cor_err):
        '''effective mass and its error based on a cosh correlation function.
        A lattice of even size is assumed.

        cor: (N/2,)
            value of  wall to wall correlation function on the first half of the lattice
        cor_err: (N/2,)
            error of correlation function on the first half of the lattice

        Returns
        m_eff: (N/2,)
            effective mass
        m_eff_err: (N/2,)
            error of the effective mass
        '''
        cor_1 = np.roll(cor, -1) # shift to d+1
        m_eff = - np.log(cor_1 / cor)
        m_eff_err = np.roll(cor_err, -1)/cor_1 - cor_err/cor 
        # m_eff_err = np.sqrt( (np.roll(ww_cor_err_mirrored, -1)/cor_1)**2 - (ww_cor_err_mirrored/cor)**2 )

        return m_eff, m_eff_err


    beta_str = str(np.round(beta, 4)).replace('.', '_')
    ds_2, cor, cor_err = np.load('data/corfuncs/beta_%s.npy'%beta_str)
    
    m_eff_cosh, m_eff_err_cosh = cosh_corfunc(cor, cor_err)
    m_eff_exp, m_eff_err_exp = exp_corfunc(cor, cor_err)

    fig = plt.figure(figsize=(8,6))

    cut = 200 # adjust manually
    plt.errorbar(ds_2[:cut], m_eff_cosh[:cut], yerr=m_eff_err_cosh[:cut], fmt='.', capsize=2, label='$\cosh$', c='red')
    plt.errorbar(ds_2[:cut]-0.2, m_eff_exp[:cut], yerr=m_eff_err_exp[:cut], fmt='.', capsize=2, label='$\exp$', c='b') # slightly shift data points to avoid overlapping
    # plt.ylim(bottom=0.1, top=0.183)
    # plt.ylim(bottom=0.3)
    # plt.ylim(top=0.625)
    plt.xlabel(r'wall separation [$a$]')
    plt.ylabel('effective mass $m_{eff}$')
    fig.gca().xaxis.set_major_locator(MaxNLocator(integer=True)) # set major ticks at integer positions only
    plt.legend(prop={'size': 12}, frameon=True)
    plt.show()
    # fig.savefig('plots/corfuncs/effective_mass/%s.pdf'%beta_str)

    return

# values of beta
# [0.6, 0.6667, 0.7333, 0.8, 0.8667, 0.9333, 1.0, 1.0667, 1.1333, 1.2, 1.2667, 1.3333, 1.4]
# effective_mass(1.4)


def adjust_fitting():
    '''Allows manual adjustment of fitting the correlation length to the processed correlation function data (normalized and mirror averaged).
    A plot with the new fitting is produced and the inferred correlation length, its error and the associated chi2 are printed.
    These can then be manually added to the data/corlen_data.txt file.
    '''
    def fit(d,xi):
        return (np.cosh((d-N_2)/xi) - 1) / (np.cosh(N_2/xi) - 1)
    
    ds, cor, cor_err = np.load('data/corfuncs/beta_1_4.npy')

    fit_upper = 220 # inclusive upper bound on wall separation to include in fitting
    N_2 = ds[-1]
    
    mask = ds <= fit_upper
    # mask = cor > 0
    popt, pcov = curve_fit(fit, ds[mask], cor[mask], sigma=cor_err[mask], absolute_sigma=True)
    cor_length = popt[0] # in units of lattice spacing
    cor_length_err = np.sqrt(pcov[0][0])

    r = cor[mask] - fit(ds[mask], *popt)
    reduced_chi2 = np.sum((r/cor_err[mask])**2) / (mask.size - 1) # dof = number of observations - number of fitted parameters

    fig = plt.figure(figsize=(8,6))

    plt.errorbar(ds, cor, yerr=cor_err, fmt='.', capsize=2, zorder=1)
    ds_fit = np.linspace(0, ds[mask][-1], 500)
    plt.plot(ds_fit, fit(ds_fit,*popt), c='g', zorder=2, label='$\\xi = %.3f \pm %.3f$\n $\chi^2/DoF = %.3f$'%(cor_length, cor_length_err, reduced_chi2))
    # plt.ylim(bottom=2e-2, top=2)
    plt.yscale('log')
    plt.xlabel(r'wall separation $d$ [$a$]')
    plt.ylabel('wall wall correlation $C_{ww}(d)$')
    plt.legend(prop={'size':12}, frameon=True, loc='upper right') # location to not conflict with error bars
    fig.gca().xaxis.set_major_locator(MaxNLocator(integer=True)) # set major ticks at integer positions only
    plt.show()

    print('corlen: %.18E \ncorlen_err: %.18E \nchi2: %.18E'%(cor_length, cor_length_err, reduced_chi2))
    

# adjust_fitting()