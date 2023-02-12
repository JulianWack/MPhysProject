import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler

from SU2xSU2 import SU2xSU2
from calibrate_paras import calibrate


plt.style.use('science')
plt.rcParams.update({'font.size': 20})
# plt.rcParams.update({'text.usetex': False}) # faster rendering
mpl.rcParams['axes.prop_cycle'] = cycler(color=['k', 'g', 'b', 'r'])


def grid_search():
    '''
    Performs grid search of the mass acceleration parameter. Conventional value is roughly equal to the inverse correlation length.
    For the considered value pair beta, N (which from the asymptotic scaling plot was found to approximate the continuum well) this is about 1/20,
    motivating the searched range and the normalization of the cost function.
    '''
    beta = 1.1333  
    N, a = 128, 1
    xi = 20

    n = 11 # total number of masses for which cost function is computed
    masses = np.zeros(n)
    masses[:-1] = np.linspace(1/1000, 1/2, n-1)
    masses[-1] = 1/xi
    times, acc_rates, chi_IATs, chi_IATs_err = np.zeros((4,n))
    cost_func, cost_func_err = np.zeros((2,n))

    for i,mass in enumerate(masses):
        model_paras = {'N':N, 'a':a, 'ell':9, 'eps':1/9, 'beta':beta, 'mass':mass}
        paras_calibrated = calibrate(model_paras, accel=True)
        sim_paras = {'M':15000, 'thin_freq':1, 'burnin_frac':0.1, 'accel':True, 'store_data':False}
        model, model_paras = calibrate(paras_calibrated, sim_paras, production_run=True)

        chi_IATs[i], chi_IATs_err[i], _, _ = model.susceptibility_IAT()
        times[i], acc_rates[i] = model.time, model.acc_rate
        cost_func[i] = times[i] / acc_rates[i] * chi_IATs[i]
        cost_func_err[i] = times[i] / acc_rates[i] * chi_IATs_err[i]

        header_str = 'acceleration masses, cost function and its error, simulation time [sec], acceptance rate, susceptibility IAT and its error'
        np.savetxt('data/accel_mass_search.txt', np.row_stack((masses, cost_func, cost_func_err, times, acc_rates, chi_IATs, chi_IATs_err)), header=header_str)
        print('%d/%d done'%(i+1,n))

    idx = np.argmin(cost_func)
    print('Value of mass parameter yielding most efficient acceleration: M = %.4f'%masses[idx])
    min_cost = cost_func[idx]
    cost_func /= min_cost
    cost_func_err /= min_cost

    fig = plt.figure(figsize=(8,6))

    plt.errorbar(masses, cost_func, yerr=cost_func_err, fmt='.', capsize=2)
    plt.xlabel('mass parameter $M$')
    plt.ylabel('cost function $L(M)/L(M^*)$')
    plt.yscale('log')
    plt.show()
    # fig.savefig('plots/mass_parameter.pdf')


grid_search()