import numpy as np
from SU2xSU2 import SU2xSU2
from calibrate_paras import calibrate

# manual parameters
model = SU2xSU2(N=16, a=1, ell=4, eps=1/4, beta=1)
model.run_HMC(200, 1, 0, accel=True, store_data=False) 

# automatic calibration
# model_paras = {'N':400, 'a':1, 'ell':20, 'eps':1/20, 'beta':1.3333}
# paras_calibrated = calibrate(model_paras, accel=True)
# print('calibration completed!')
# sim_paras = {'M':2000, 'thin_freq':1, 'burnin_frac':0.2, 'accel':True, 'store_data':True}
# model, paras = calibrate(paras_calibrated, sim_paras, production_run=True)
# print('Parameters used during production run: ',paras)

# load stored simulation
# paras = np.loadtxt('data/single_run/model_paras.txt')
# sim_paras = np.loadtxt('data/single_run/sim_paras.txt')
# print('Loading simulation:\nN, a, ell, eps, beta\n',paras,'\nM, thin freq, burn in, accept rate\n',sim_paras)
# model = SU2xSU2(*paras)
# model.load_data()

# # apply additional thinning
# breakpoint()
# model.configs = model.configs[-1].reshape((1, model.N, model.N, 4))
# model.sweeps = model.sweeps[-1]
# model.M = model.configs.shape[0]


# exp__dH_avg , exp__dH_err = model.exp__dH(make_plot=True)
# print('<exp(-dH)> = %.5f +/- %.5f \n'%(exp__dH_avg, exp__dH_err))

# m_avg , m_err = model.order_parameter(make_plot=True)
# for i,(avg, err) in enumerate(zip(m_avg, m_err)):
#         print('<m_%d> : %.5f +/- %.5f'%(i, avg, err))

# s_avg, s_err, IAT_s, IAT_s_err = model.internal_energy_density() 
# print('action per site = %.5f +/- %.5f'%(s_avg, s_err))
# print('IAT = %.5f +/- %.5f \n'%(IAT_s, IAT_s_err))

# cor_len, cor_len_err, _ = model.ww_correlation(make_plot=True)
# print('correlation length = %.5f +/- %.5f \n'%(cor_len, cor_len_err))