import numpy as np
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler

plt.style.use('science')
plt.rcParams.update({'font.size': 20})
# plt.rcParams.update({'text.usetex': False}) # faster rendering
mpl.rcParams['axes.prop_cycle'] = cycler(color=['k', 'g', 'b', 'r'])


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


def effective_mass(N, beta):
    '''
    Effective mass plot for passed value pair of lattice size N (assumed even) and beta. This tends to highlight the worst in the data.
    Loads in correlation function results for one value pair of N and beta and produces an effective mass plot.
    The effective mass will be computed based on the assumption that the correlation function follows the shape of
    a cosh (analytically expected due to periodic boundary conditions) or of a pure exponential decay with the later tending
    to produce a better behaved plot.
    '''
    beta_str = str(np.round(beta, 4)).replace('.', '_')
    ds, ww_cor, ww_cor_err = np.load('data/corfuncs/beta_%s.npy'%beta_str)

    N_2 = int(N/2)
    ds_2 = ds[:N_2+1]
    # exploit symmetry about N/2 to reduce errors (effectively increasing number of data points by factor of 2)
    cor = 1/2 * (ww_cor[:N_2+1] + ww_cor[N_2:][::-1])
    cor_err = np.sqrt(ww_cor_err[:N_2+1]**2 + ww_cor_err[N_2::-1]**2)

    m_eff_cosh, m_eff_err_cosh = cosh_corfunc(cor, cor_err)
    m_eff_exp, m_eff_err_exp = exp_corfunc(cor, cor_err)

    fig = plt.figure(figsize=(8,6))

    cut = 22 # adjust manually
    plt.errorbar(ds_2[:cut], m_eff_cosh[:cut], yerr=m_eff_err_cosh[:cut], fmt='.', capsize=2, label='$\cosh$', c='red')
    plt.errorbar(ds_2[:cut]-0.2, m_eff_exp[:cut], yerr=m_eff_err_exp[:cut], fmt='.', capsize=2, label='$\exp$', c='b') # slightly shift data points to avoid overlapping
    plt.xlabel(r'wall separation [$a$]')
    plt.ylabel('effective mass')
    fig.gca().xaxis.set_major_locator(MaxNLocator(integer=True)) # set major ticks at integer positions only
    plt.legend(prop={'size': 12}, frameon=True)
    plt.show()
    # fig.savefig('plots/corfuncs/effective_mass/%s.pdf'%beta_str)

    return


effective_mass(64, 0.8667)