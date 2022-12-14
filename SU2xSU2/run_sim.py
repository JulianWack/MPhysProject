from SU2xSU2 import SU2xSU2


model = SU2xSU2(N=16, a=1, ell=10, eps=0.1, beta=1)
model.run_HMC(5000, 100, 0.1, store_data=False)    


# exp__dH_avg , exp__dH_err = model.exp__dH(make_plot=False)
# print('<exp(-dH)> = %.5f +/- %.5f \n'%(exp__dH_avg, exp__dH_err))

# m_avg , m_err = model.order_parameter(make_plot=True)
# for i,(avg, err) in enumerate(zip(m_avg, m_err)):
#         print('<m_%d> : %.5f +/- %.5f'%(i, avg, err))

# s_avg, s_err, IAT_s, IAT_s_err = model.action_per_site() 
# print('action per site = %.5f +/- %.5f'%(s_avg, s_err))
# print('IAT = %.5f +/- %.5f \n'%(IAT_s, IAT_s_err))

# chi_avg, chi_err, IAT_chi, IAT_chi_err = model.susceptibility_per_site() 
# print('susceptibility per site = %.5f +/- %.5f'%(chi_avg, chi_err))
# print('IAT = %.5f +/- %.5f \n'%(IAT_chi, IAT_chi_err))

# c_avg, c_err, IAT_c, IAT_c_err = model.specific_heat_per_site() 
# print('specific heat per site = %.5f +/- %.5f'%(c_avg, c_err))
# print('IAT = %.5f +/- %.5f'%(IAT_c, IAT_c_err))