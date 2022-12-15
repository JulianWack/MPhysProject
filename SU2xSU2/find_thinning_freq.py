import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler

from SU2xSU2 import SU2xSU2


plt.style.use('science')
plt.rcParams.update({'font.size': 20})
# plt.rcParams.update({'text.usetex': False}) # faster rendering
mpl.rcParams['axes.prop_cycle'] = cycler(color=['k', 'g', 'b', 'r'])

thin_freqs = np.logspace(np.log10(2), 2, 10, dtype=int)
s_IAT = np.zeros_like(thin_freqs, dtype=float)
s_IAT_err = np.zeros_like(thin_freqs, dtype=float)
c_IAT = np.zeros_like(thin_freqs, dtype=float)
c_IAT_err = np.zeros_like(thin_freqs, dtype=float)

model = SU2xSU2(N=16, a=1, ell=10, eps=0.1, beta=1)
# # runs ingle simulation and store unthinned chain 
# model.run_HMC(20000, 1, 0.1, store_data=True) 
# # or load stored simulation
paras = np.load('data/sim_paras.npy')
model = SU2xSU2(*paras)
model.load_data()

unthinned_configs = model.configs

for i,thin_freq in enumerate(thin_freqs):
    thinned_configs = unthinned_configs[::thin_freq]
    model.configs = thinned_configs
    model.M = thinned_configs.shape[0]
    _,_, s_IAT[i], s_IAT_err[i] = model.action_per_site()
    _,_, c_IAT[i], c_IAT_err[i] = model.specific_heat_per_site()


fig = plt.figure(figsize=(8,6))

plt.errorbar(thin_freqs, s_IAT, yerr=s_IAT_err, fmt='x', capsize=2, color='k', label='action per site')
plt.plot(thin_freqs, s_IAT, color='k')
plt.errorbar(thin_freqs, c_IAT, yerr=c_IAT_err, fmt='x', capsize=2, color='g', label='specific heat per site')
plt.plot(thin_freqs, c_IAT, color='g')
plt.xscale('log')
plt.xlabel('thinning frequency')
plt.ylabel(r'$\tau$')
plt.legend(prop={'size': 12})

plt.show()
# fig.savefig('plots/thin_freq.pdf)