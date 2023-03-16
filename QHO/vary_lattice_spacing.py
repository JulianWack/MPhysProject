import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler

from Oscillator import Oscillator
from calibrate_paras import calibrate

plt.style.use('science')
plt.rcParams.update({'font.size': 20})
# plt.rcParams.update({'text.usetex': False}) # faster rendering
mpl.rcParams['axes.prop_cycle'] = cycler(color=['k', 'g', 'b', 'r'])


m = 1
w = 1
N = 100

# num = 15
# a_s = np.linspace(1, 0.05, num)
# focus on region with interesting behavior 
small = np.linspace(0.05, 0.4, 10)
large = np.linspace(0.45, 1, 4)
a_s = np.concatenate((small,large))[::-1]
num = a_s.size

paras_record = np.empty((num, 6)) # each row stores basic simulation parameters and results

# x2, x2_err, x2_lat, x2_cts
x2_data = np.zeros((4, num))
x2_data[-1] = np.full(num, 1/(2*m*w))

# E0, E0_err, E0_lat, E0_cts
E0_data = np.zeros((4, num))
E0_data[-1] = np.full(num, 0.5*w)

# energy gap: dE, dE_err, dE_lat, dE_cts
dE_data = np.zeros((4, num))
dE_data[-1] = np.full(num, w)


prev_ell, prev_eps = 2, 1/2
for i, a in enumerate(a_s):
    model_paras = {'m':m, 'w':w, 'N':N, 'a':a, 'ell':prev_ell, 'eps':prev_eps}
    paras_calibrated = calibrate(model_paras, accel=True)
    prev_ell, prev_eps = paras_calibrated['ell'], paras_calibrated['eps']

    QHO = Oscillator(**paras_calibrated)
    sim_paras = {'M':50000, 'thin_freq':1, 'burnin_frac':0.1, 'accel':True, 'store_data':False}
    QHO.run_HMC(**sim_paras) 

    # measurements
    tau_int, tau_int_err = QHO.autocorrelation(make_plot=False)
    paras_record[i] = np.array([a, QHO.ell, QHO.eps, QHO.acc_rate, tau_int, tau_int_err])

    x2_data[0,i], x2_data[1,i] = QHO.x_moment(2)
    x2_data[2,i] = QHO.x2_dis_theo()

    E0_data[0,i], E0_data[1,i] = QHO.gs_energy()
    E0_data[2,i] = QHO.gs_energy_dis_theo()

    dE_data[2,i] = QHO.delta_E_dis_theo()
    dE_data[0,i], dE_data[1,i] = QHO.correlation(upper=dE_data[2,i]/a) # use analytically known correlation length as guide to find fitting range
    

    # save partial results
    np.savetxt('data/vary_a/paras.txt', paras_record, header='a, ell, eps, acc_rate, tau_int, tau_int_err')
    np.save('data/vary_a/x2_data_new.npy', x2_data)
    np.save('data/vary_a/E0_data_new.npy', E0_data)
    np.save('data/vary_a/dE_data_new.npy', dE_data)

    print('-'*32)
    print('Completed %d/%d: a=%.3f with acceptance rate %.2f%%'%(i+1, num, a, QHO.acc_rate*100))
    print('-'*32)


# make plots 

# x^2 plot
fig = plt.figure(figsize=(8,6))
plt.errorbar(a_s, x2_data[0], yerr=x2_data[1], fmt='.', capsize=2, label='HMC')
plt.plot(a_s, x2_data[2], label='lattice')
plt.plot(a_s, x2_data[3], label='continuum')
plt.xlabel('lattice spacing $a$')
plt.ylabel(r'$\langle x^2 \rangle$')
plt.legend(prop={'size': 12}, frameon=True)
plt.show()
# fig.savefig('plots/vary_lattice_spacing/x2_vs_a.pdf')

# E0 plot
fig = plt.figure(figsize=(8,6))
plt.errorbar(a_s, E0_data[0], yerr=E0_data[1], fmt='.', capsize=2, label='HMC')
plt.plot(a_s, E0_data[2], label='lattice')
plt.plot(a_s, E0_data[3], label='continuum')
plt.xlabel('lattice spacing $a$')
plt.ylabel(r'$E_0$')
plt.legend(prop={'size': 12}, frameon=True)
plt.show()
# fig.savefig('plots/vary_lattice_spacing/E0_vs_a.pdf')

# E1-E0 plot
fig = plt.figure(figsize=(8,6))
plt.errorbar(a_s, dE_data[0], yerr=dE_data[1], fmt='.', capsize=2, label='HMC')
plt.plot(a_s, dE_data[2], label='lattice')
plt.plot(a_s, dE_data[3], label='continuum')
plt.xlabel('lattice spacing $a$')
plt.ylabel(r'$E_1 - E_0$')
plt.legend(prop={'size': 12}, frameon=True)
plt.show()
# fig.savefig('plots/vary_lattice_spacing/deltaE_vs_a.pdf')