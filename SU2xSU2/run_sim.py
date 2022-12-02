from SU2xSU2 import SU2xSU2


model = SU2xSU2(N=16, a=1, ell=10, eps=0.1, beta=1)
model.run_HMC(2000, 1, 0.1, store_data=False)    


exp__dH_avg , exp__dH_err = model.exp__dH(make_plot=True)
print('<exp(-dH)> = %.5f +/- %.5f '%(exp__dH_avg, exp__dH_err))