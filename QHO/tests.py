# file containing additional tests conducted during the development
# some simple plots are produced as well

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
from datetime import timedelta
from cycler import cycler
from Oscillator import Oscillator

plt.style.use('science')
plt.rcParams.update({'font.size': 20})
# plt.rcParams.update({'text.usetex': False}) # faster rendering
mpl.rcParams['axes.prop_cycle'] = cycler(color=['k', 'g', 'b', 'r']) 


##### equipartition test ####
def equipartiton_test():
    m = 1
    w = 1
    N = 4
    a = 1
    ell = 10
    eps = 0.1

    kins = []

    QHO = Oscillator(m, w, N, a, ell, eps)
    k = np.arange(0, N)
    n_iter = 10000
    for i in range(n_iter):
        ps = QHO.p_samples()
        A = (m/a * (4*np.sin(np.pi * k/N)**2 + (a*w)**2) )**(-1)
        K = 0.5/N * np.sum( np.abs(np.fft.fft(ps))**2 * A)
        kins.append(K/N)

    print(np.mean(kins), np.std(kins)/np.sqrt(n_iter))

# equipartiton_test()
######


###### ACF comparison ######
def ACF_compare():
    m = 1
    w = 1
    N = 100
    a = 1
    ell = 4
    eps = 0.25

    Ms = np.linspace(500, 1000, 2) # number of HMC trajectories
    ACF_residuals = np.zeros_like(Ms)
    IAT_residuals = np.zeros_like(Ms)
    times_naive, times_fast = np.zeros_like(Ms), np.zeros_like(Ms)

    for i,M in enumerate(Ms):
        QHO = Oscillator(m, w, N, a, ell, eps)
        QHO.run_HMC(int(M), 1, 0.1, accel=False, store_data=False)
        data = QHO.xs
        Ms[i] = data.shape[0] # burn in reduces number of samples between we need to find the correlation

        _, ACF, ACF_err, IAT, IAT_err, dt = QHO.correlator(data, my_upper=data.shape[0])
        _, ACF_fast, ACF_err_fast, IAT_fast, IAT_err_fast, dt_fast = QHO.correlator_fast(data)

        ACF_residuals[i] = np.sum(np.abs(ACF-ACF_fast))/np.sum(ACF)
        IAT_residuals[i] = (IAT - IAT_fast) / IAT
        times_naive[i], times_fast[i] = dt, dt_fast


    fig, (ax_time, ax_IAT_diff) =  plt.subplots(1, 2, figsize=(14,6))
    # fig, (ax_time, ax_ACF_diff, ax_IAT_diff) =  plt.subplots(1, 3, figsize=(20,6))

    ax_time.plot(Ms, times_naive, label='Naive sum')
    ax_time.plot(Ms, times_fast, label='FFT approach')
    ax_time.set_xlabel('\# samples to correlate')
    ax_time.set_ylabel('CPU time [sec]')
    ax_time.set_yscale('log')
    ax_time.legend(prop={'size': 12})

    # ax_ACF_diff.plot(Ms, ACF_residuals)
    # ax_ACF_diff.set_xlabel('\# samples to correlate')
    # ax_ACF_diff.set_ylabel('$ACF$ residual')

    ax_IAT_diff.errorbar(Ms, 100*IAT_residuals)
    ax_IAT_diff.set_xlabel('\# samples to correlate')
    ax_IAT_diff.set_ylabel('\% difference in $IAT$')


    fig.tight_layout()
    fig.savefig('plots/ACF_comparison.pdf')
    plt.show()

# ACF_compare()
######