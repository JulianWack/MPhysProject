import numpy as np
from Oscillator import Oscillator


m = 1
w = 1
N = 100
a = 0.01 
ell = 10  
eps = 1/ell * 0.25

# QHO = Oscillator(m, w, N, a, ell, eps)
# QHO.run_HMC(10000, 1, 0.0, store_data=True)


paras = np.load('data/sim_paras.npy')
QHO = Oscillator(*paras)
QHO.load_data()


# tau_int, tau_int_err = QHO.autocorrelation(make_plot=False)
# print('Configuration tau_int =  %.5f +/- %.5f'%(tau_int, tau_int_err))

# QHO.show_one_configuration()
# QHO.x_moment(1,make_plot=True)
x2, x2_err = QHO.x_moment(2,make_plot=True)
# QHO.exp__dH(make_plot=True)
# E0, E0_err = QHO.gs_energy()
# QHO.plot_wavefunction(100)
# tau_int, tau_int_err = QHO.autocorrelation(make_plot=True)
# print('Configuration tau_int =  %.5f +/- %.5f'%(tau_int, tau_int_err))
# dE, dE_err = QHO.correlation(make_plot=True)

# x2_dis = QHO.x2_dis_theo()
# print('Difference to true <x^2>: %.5f +/- %.5f'%(x2-x2_dis, x2_err))
# E0_dis = QHO.gs_energy_dis_theo()
# print('Difference to true E_0: %.5f +/- %.5f'%(E0-E0_dis, E0_err))
# dE_dis = QHO.delta_E_dis_theo()
# print('Difference to true E_1-E_0: %.5f +/- %.5f'%(dE-dE_dis, dE_err))
# print('mine: ', dE)