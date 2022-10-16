import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
from Oscillator import Oscillator

plt.style.use('science')
plt.rcParams.update({'font.size': 20})
# plt.rcParams.update({'text.usetex': False}) # faster rendering
mpl.rcParams['axes.prop_cycle'] = cycler(color=['k', 'g', 'b', 'r'])


m = 1
w = 1
N = 100

num = 30
a_s = np.linspace(0.05, 1, num)
eps_guess = np.linspace(0.1, 0.3, num)
ell_guess = np.rint(1/eps_guess)

paras_record = np.empty((num, 6)) # each row stores basic simulation parameters and results

x2s, x2_errs = np.full(num, np.nan), np.full(num, np.nan)
x2s_dis = np.full(num, np.nan)
x2s_cts = np.full(num, 1/(2*m*w))

E0s, E0_errs = np.full(num, np.nan), np.full(num, np.nan)
E0s_dis = np.full(num, np.nan)
E0s_cts = np.full(num, 0.5*w)

E1_E0s, E1_E0_errs = np.full(num, np.nan), np.full(num, np.nan)
E1_E0s_dis = np.full(num, np.nan)
E1_E0s_cts = np.full(num, w) 

for i, a in enumerate(a_s):
    ell = int(ell_guess[i])
    eps = eps_guess[i]
    good_acc_rate = False
    while good_acc_rate == False:
        QHO = Oscillator(m, w, N, a, ell, eps)
        QHO.run_HMC(50000, 15, 0.1, store_data=False)
        acc_rate = QHO.acc_rate
        # if acceptance rate outwith desired range, adjust step size by +/- 10% and run again 
        if acc_rate < 0.6:
            eps *= 0.9
            ell = int(1/eps)
        elif acc_rate > 0.8:
            eps *= 1.1
            ell = int(1/eps)
        else:
            good_acc_rate = True


    tau_int, _ = QHO.autocorrelation(make_plot=False)
    paras_record[i] = np.array([a, ell, eps, ell*eps, acc_rate, tau_int])
    print('-'*32)
    print('Completed %d: a=%.3f with eps=%.3f at %.2f%% and tau=%.3f'%(i, a, eps, acc_rate*100, tau_int))
    print('-'*32)


    x2s[i], x2_errs[i] = QHO.x_moment(2,make_plot=False)
    x2s_dis[i] = QHO.x2_dis_theo()

    E0s[i], E0_errs[i] = QHO.gs_energy()
    E0s_dis[i] = QHO.gs_energy_dis_theo()

    E1_E0s[i], E1_E0_errs[i] = QHO.correlation(make_plot=False)
    E1_E0s_dis[i] = QHO.delta_E_dis_theo()


# store data
np.savetxt('data/vary_a/paras', paras_record, header='a, ell, eps, ell*eps, acc_rate, tau_int')

np.save('data/vary_a/x2', x2s)
np.save('data/vary_a/x2_err', x2_errs)
np.save('data/vary_a/x2_dis', x2s_dis)
np.save('data/vary_a/x2_cts', x2s_cts)

np.save('data/vary_a/E0', E0s)
np.save('data/vary_a/E0_err', E0_errs)
np.save('data/vary_a/E0_dis', E0s_dis)
np.save('data/vary_a/E0_cts', E0s_cts)

np.save('data/vary_a/E1_E0', E1_E0s)
np.save('data/vary_a/E1_E0_err', E1_E0_errs)
np.save('data/vary_a/E1_E0_dis', E1_E0s_dis)
np.save('data/vary_a/E1_E0_cts', E1_E0s_cts)


# x^2 plot
fig = plt.figure(figsize=(8,8))
plt.errorbar(a_s, x2s, yerr=x2_errs, fmt='x', capsize=2, label='HMC')
plt.plot(a_s, x2s_dis, label='dis theory')
plt.plot(a_s, x2s_cts, label='cts theory')
plt.xlabel('a')
plt.ylabel(r'$\langle x^2 \rangle$')
plt.legend(prop={'size': 12})
# plt.show()
fig.savefig('plots/vary_lattice_spacing/x2_vs_a.pdf')

# E0 plot
fig = plt.figure(figsize=(8,8))
plt.errorbar(a_s, E0s, yerr=E0_errs, fmt='x', capsize=2, label='HMC')
plt.plot(a_s, E0s_dis, label='dis theory')
plt.plot(a_s, E0s_cts, label='cts theory')
plt.xlabel('a')
plt.ylabel(r'$E_0$')
plt.legend(prop={'size': 12})
# plt.show()
fig.savefig('plots/vary_lattice_spacing/E0_vs_a.pdf')

# E1-E0 plot
fig = plt.figure(figsize=(8,8))
plt.errorbar(a_s, E1_E0s, yerr=E1_E0_errs, fmt='x', capsize=2, label='HMC')
plt.plot(a_s, E1_E0s_dis, label='dis theory')
plt.plot(a_s, E1_E0s_cts, label='cts theory')
plt.xlabel('a')
plt.ylabel(r'$E_1 - E_0$')
plt.legend(prop={'size': 12})
# plt.show()
fig.savefig('plots/vary_lattice_spacing/deltaE_vs_a.pdf')