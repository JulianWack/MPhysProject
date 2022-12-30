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

thin_freqs = np.logspace(np.log10(2), 2, 10, dtype=int)
e_IAT = np.zeros_like(thin_freqs, dtype=float)
e_IAT_err = np.zeros_like(thin_freqs, dtype=float)

# # run single simulation and store unthinned chain
model_paras = {'N':32, 'a':1, 'ell':3, 'eps':1/3, 'beta':2}
paras_calibrated = calibrate(model_paras)
print('calibration completed!')
sim_paras = {'M':2000, 'thin_freq':1, 'burnin_frac':0.1, 'store_data':False}
model, paras = calibrate(paras_calibrated, sim_paras, production_run=True)
unthinned_configs = model.configs
print('Parameters used during production run: ',paras)

# # or load stored simulation
# paras = np.loadtxt('data/model_paras.txt')
# model = SU2xSU2(*paras)
# model.load_data()
# unthinned_configs = model.configs


for i,thin_freq in enumerate(thin_freqs):
    thinned_configs = unthinned_configs[::thin_freq]
    model.configs = thinned_configs
    model.M = thinned_configs.shape[0]
    _,_, e_IAT[i], e_IAT_err[i] = model.internal_energy_density()


fig = plt.figure(figsize=(8,6))

plt.errorbar(thin_freqs, e_IAT, yerr=e_IAT_err, fmt='x', capsize=2, color='k')
plt.plot(thin_freqs, e_IAT, color='k')
plt.xscale('log')
plt.xlabel('thinning frequency')
plt.ylabel(r'$\tau$')

plt.show()
# fig.savefig('plots/thin_freq.pdf)