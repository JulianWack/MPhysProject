import numpy as np
from Oscillator import Oscillator


m = 1
w = 1
N = 100
a = 1 
ell = 50 #25 
eps = 0.4 #0.35 

# QHO = Oscillator(m, w, N, a, ell, eps)
# QHO.run_HMC(100000, 10, 0.1, store_data=True)

paras = np.load('data/sim_paras.npy')
QHO = Oscillator(*paras)
QHO.load_data()


# QHO.show_one_configuration()
# QHO.x_moment(1,make_plot=True)
x2_MC, x2_MC_err = QHO.x_moment(2,make_plot=False)
# QHO.exp__dH(make_plot=True)
E0_MC, E0_MC_err = QHO.gs_energy()
# QHO.plot_wavefunction(50)
# QHO.correlation_configs(make_plot=True)
# QHO.autocorrelation_latticesites(make_plot=True)

x2_dis = QHO.x2_dis_theo()
print('Difference to true <x^2>: %.5f +/- %.5f'%(x2_MC-x2_dis, x2_MC_err))
E0_dis = QHO.gs_energy_dis_theo()
print('Difference to true E_0: %.5f +/- %.5f'%(E0_MC-E0_dis, E0_MC_err))
