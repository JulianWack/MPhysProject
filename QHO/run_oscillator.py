import numpy as np
from Oscillator import Oscillator
from calibrate_paras import calibrate


m = 1
w = 1
N = 100
a = 1
ell = 8
eps = 1/8


# calibrate
# model_paras = {'m':m, 'w':w, 'N':N, 'a':a, 'ell':ell, 'eps':eps}
# paras_calibrated = calibrate(model_paras, accel=False, traj_length=1)
# print(paras_calibrated)
# QHO = Oscillator(**paras_calibrated)
# sim_paras = {'M':10000, 'thin_freq':1, 'burnin_frac':0.1, 'accel':False, 'store_data':False}
# QHO.run_HMC(**sim_paras) 

# manual run
# QHO = Oscillator(m, w, N, a, ell, eps)
# QHO.run_HMC(10000, 1, 0.1, accel=False, store_data=True)

# load stored run
# paras = np.load('data/sim_paras.npy')
# QHO = Oscillator(*paras)
# QHO.load_data()


# QHO.show_one_configuration()
# QHO.x_moment(1,make_plot=True)
# x2, x2_err = QHO.x_moment(2, make_plot=True)
# QHO.exp__dH(make_plot=True)
# E0, E0_err = QHO.gs_energy()
# QHO.plot_wavefunction(100)
# tau_int, tau_int_err = QHO.autocorrelation(make_plot=False)
# print('Configuration tau_int =  %.5f +/- %.5f'%(tau_int, tau_int_err))
# dE, dE_err = QHO.correlation(upper=20, make_plot=True)

# x2_dis = QHO.x2_dis_theo()
# print('Difference to discrete theory <x^2>: %.5f +/- %.5f'%(x2-x2_dis, x2_err))
# E0_dis = QHO.gs_energy_dis_theo()
# print('Difference to discrete theory E_0: %.5f +/- %.5f'%(E0-E0_dis, E0_err))
# dE_dis = QHO.delta_E_dis_theo()
# print('Difference to discrete theory E_1-E_0: %.5f +/- %.5f'%(dE-dE_dis, dE_err))