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
# a_s = np.linspace(0.05, 1, num)
# eps_guess = np.linspace(0.1, 0.3, num)
# ell_guess = np.rint(1/eps_guess)

# when already run for the set lattice spacings
a_s = np.loadtxt('data/vary_a/paras')[:,0]
eps_guess = np.loadtxt('data/vary_a/paras')[:,2]
ell_guess = np.rint(np.loadtxt('data/vary_a/paras')[:,1])

paras_record = np.empty((num, 7)) # each row stores basic simulation parameters and results

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
        QHO.run_HMC(100000, 15, 0.1, store_data=False)
        acc_rate = QHO.acc_rate
        # if acceptance rate outwith desired range, adjust step size by +/- 10% and run again 
        if acc_rate < 0.6:
            eps *= 0.9
            ell = int(1/eps)
        elif acc_rate > 0.75:
            eps *= 1.1
            ell = int(1/eps)
        else:
            good_acc_rate = True


    tau_int, tau_int_err = QHO.autocorrelation(make_plot=False)
    paras_record[i] = np.array([a, ell, eps, ell*eps, acc_rate, tau_int, tau_int_err])
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
np.savetxt('data/vary_a/paras', paras_record, header='a, ell, eps, ell*eps, acc_rate, tau_int, tau_int_err')

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


# make plots 
# simulation parameters
fig, (ax_traj_len, ax_acc_rate, ax_tau_int) =  plt.subplots(1, 3, figsize=(24,6))

ax_traj_len.scatter(a_s, paras_record[:,3], marker='x')
ax_traj_len.plot(a_s, paras_record[:,3])
ax_traj_len.set_xlabel('a')
ax_traj_len.set_ylabel('$\epsilon \ell$')

ax_acc_rate.scatter(a_s, paras_record[:,4]*100, marker='x')
ax_acc_rate.plot(a_s, paras_record[:,4]*100)
ax_acc_rate.set_xlabel('a')
ax_acc_rate.set_ylabel('acceptance rate in \%')

ax_tau_int.errorbar(a_s, paras_record[:,5], yerr=paras_record[:,6], fmt='x', capsize=2)
ax_tau_int.set_xlabel('a')
ax_tau_int.set_ylabel(r'$\tau_{int}$')

fig.tight_layout()
# plt.show()
fig.savefig('plots/vary_lattice_spacing/parameters.pdf')


# x^2 plot
fig = plt.figure(figsize=(8,6))
plt.errorbar(a_s, x2s, yerr=x2_errs, fmt='x', capsize=2, label='HMC')
plt.plot(a_s, x2s_dis, label='dis theory')
plt.plot(a_s, x2s_cts, label='cts theory')
plt.xlabel('a')
plt.ylabel(r'$\langle x^2 \rangle$')
plt.legend(prop={'size': 12})
# plt.show()
fig.savefig('plots/vary_lattice_spacing/x2_vs_a.pdf')

# E0 plot
fig = plt.figure(figsize=(8,6))
plt.errorbar(a_s, E0s, yerr=E0_errs, fmt='x', capsize=2, label='HMC')
plt.plot(a_s, E0s_dis, label='dis theory')
plt.plot(a_s, E0s_cts, label='cts theory')
plt.xlabel('a')
plt.ylabel(r'$E_0$')
plt.legend(prop={'size': 12})
# plt.show()
fig.savefig('plots/vary_lattice_spacing/E0_vs_a.pdf')

# E1-E0 plot
fig = plt.figure(figsize=(8,6))
plt.errorbar(a_s, E1_E0s, yerr=E1_E0_errs, fmt='x', capsize=2, label='HMC')
plt.plot(a_s, E1_E0s_dis, label='dis theory')
plt.plot(a_s, E1_E0s_cts, label='cts theory')
plt.xlabel('a')
plt.ylabel(r'$E_1 - E_0$')
plt.legend(prop={'size': 12})
# plt.show()
fig.savefig('plots/vary_lattice_spacing/deltaE_vs_a.pdf')