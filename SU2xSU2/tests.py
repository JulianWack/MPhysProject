import numpy as np
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
from astropy.stats import jackknife_stats
from scipy.optimize import curve_fit
from alive_progress import alive_bar
import timeit
import time
from datetime import timedelta

plt.style.use('science')
plt.rcParams.update({'font.size': 20})
# plt.rcParams.update({'text.usetex': False}) # faster rendering
mpl.rcParams['axes.prop_cycle'] = cycler(color=['k', 'g', 'b', 'r'])

import SU2_mat_routines as SU2
from SU2xSU2 import SU2xSU2
from calibrate_paras import calibrate
import correlations


##### Check if my matrix routine gives same result as np #####
def test_SU2_routines():
    # Define a N by N lattice and and describe matrices at every site by some arbitrary parameters
    N = 10
    # np.random.seed(42)
    aa = 10*np.random.random((N,N,4))
    bb = -66.8*np.random.random((N,N,4))

    # compute quantities to compare using my routines
    my_A = aa
    my_B = bb
    my_C = SU2.dot(my_A, my_B)

    # compute quantities to compare using np routines
    def make_np_mat(a):
        return np.matrix( [[a[0]+1j*a[3], a[2]+1j*a[1]], [-a[2]+1j*a[1], a[0]-1j*a[3]]] )

    np_A = np.empty((N,N), dtype=object)
    np_B = np.empty((N,N), dtype=object)
    np_C = np.empty((N,N), dtype=object)

    for i in range(N):
        for j in range(N):
            np_A[i,j] = make_np_mat(aa[i,j,:])
            np_B[i,j] = make_np_mat(bb[i,j,:])
            np_C[i,j] = np.matmul(np_A[i,j], np_B[i,j])


    # compare results: need to convert np.matrix object into ndarray (via .A) for allclose comparison 
    # sum 
    su2_element, k = SU2.sum(my_A, my_B)
    my_sum = SU2.make_mats(k*su2_element)
    all_equal = True
    for i in range(N):
        for j in range(N):
            same = np.allclose(my_sum[i,j].A, (np_A[i,j]+np_B[i,j]).A)
            if not same:
                all_equal = False
                print('Unequal sum at site: (%d, %d)'%(i,j))
    if all_equal:
        print('All sums equal')

    # product
    my_prod = SU2.make_mats(my_C)
    all_equal = True
    for i in range(N):
        for j in range(N):
            same = np.allclose(my_prod[i,j], np_C[i,j])
            if not same:
                all_equal = False
                print('Unequal product at site: (%d, %d)'%(i,j))
    if all_equal:
        print('All products equal')

    # hermitian conjugate
    my_hc = SU2.make_mats(SU2.hc(my_A))
    all_equal = True
    for i in range(N):
        for j in range(N):
            same = np.allclose(my_hc[i,j].A, (np_A[i,j].H).A)
            if not same:
                all_equal = False
                print('Unequal hc at site: (%d, %d)'%(i,j))
    if all_equal:
        print('All hc equal')

    # det
    my_det = SU2.det(my_A)
    all_equal = True
    for i in range(N):
        for j in range(N):
            same = np.allclose(my_det[i,j], np.linalg.det(np_A[i,j].A).real)
            if not same:
                all_equal = False
                print('Unequal det at site: (%d, %d)'%(i,j))
    if all_equal:
        print('All det equal')

    # trace
    my_tr = SU2.tr(my_A)
    all_equal = True
    for i in range(N):
        for j in range(N):
            same = np.allclose(my_tr[i,j], np.trace(np_A[i,j].A).real)
            if not same:
                all_equal = False
                print('Unequal tr at site: (%d, %d)'%(i,j))
    if all_equal:
        print('All tr equal')

# test_SU2_routines()
########


##### Compare speed of single matrix multiplication #####
def test_SU2_prod_speed():
    # tabbing important to assure that string has no tabs. Otherwise timeit throws an error
    # when using timeit outside a function, can tab all lines to the same height

    set_up = '''
import numpy as np
import SU2_mat_routines as SU2
'''

    my_test_code = ''' 
aa = np.random.random((1,1,4))
bb = np.random.random((1,1,4))

my_A = aa
my_B = bb

SU2.dot(my_A, my_B)
'''

    np_test_code = '''
aa = np.random.random(4)
bb = np.random.random(4)

def make_np_mat(a):
    return np.matrix( [[a[0]+1j*a[3], a[2]+1j*a[1]], [-a[2]+1j*a[1], a[0]-1j*a[3]]] )

np_A = make_np_mat(aa)
np_B = make_np_mat(bb)

np.matmul(np_A, np_B)
'''

    n_iter = 10000
    # print total time needed to perform n_iter executions of the test code
    print('My product: ', timeit.timeit(setup=set_up, stmt=my_test_code, number=n_iter))
    print('np product: ', timeit.timeit(setup=set_up, stmt=np_test_code, number=n_iter))

# test_SU2_prod_speed()
########


##### Check that my nearest neighbor construction and indexing works #####
def test_NN():
    N = 4 # number of lattice sites
    M = 2 # For readability, suppose matrix at each lattice site is described by M=2 parameters
    phi = np.empty((N,N,M))

    # asign arbitary parameter values
    phi[:,:,0] = np.arange(N**2).reshape((N,N))
    phi[:,:,1] = -np.arange(N**2).reshape((N,N))

    # make a (N,N,2) array storing the row and col indices of each lattice sites
    grid = np.indices((N,N)) # shape (2,N,N)
    lattice_coords = grid.transpose(1,2,0) # turns axis i of grid into axis j of idxs when axis i listed at position j; shape (N,N,2)

    # shift lattice coordinates by 1 such that the coordinates at (i,j) are those of the right, left, top, and bottom neighbor of lattice site (i,j); shape (N,N,2)
    # rolling axis=1 by -1 means all columns are moved one step to the left with periodic bcs. Hence value of resulting array at (i,j) is (i,j+1), i.e the coordinates of the right neighbor.
    right_n = np.roll(lattice_coords, -1, axis=1)
    left_n = np.roll(lattice_coords, 1, axis=1)
    top_n = np.roll(lattice_coords, 1, axis=0)
    bottom_n = np.roll(lattice_coords, -1, axis=0)

    # for each lattice site, for each neighbor, store row and col coord
    NN = np.empty((N,N,4,2), dtype=int)
    NN[:,:,0,:] = right_n # row and col indices of right neighbors
    NN[:,:,1,:] = left_n
    NN[:,:,2,:] = top_n
    NN[:,:,3,:] = bottom_n

    # make mask to apply to phi and get NN parameters
    # separate the row and column neighbor coordinates for each lattice site to use in indexing of phi
    # (N,N,4,1): all x-sites, all y-sites, all neighbors, only row coords or only col coords
    NN_rows = NN[:,:,:,0]
    NN_cols = NN[:,:,:,1]
    NN_mask = (NN_rows, NN_cols)

    # single test: 
    # find matrix parameters of the neighbors of site (0,0)
    one_neighbor_paras = phi[NN_mask][0,0]
    print(one_neighbor_paras)

    # full test:
    # find matrix parameters of the neighbors of site every lattice site (gives 4 sets of parameters for every site)
    all_neighbor_paras = phi[NN_mask] # (N,N,4,# paras)
    print(all_neighbor_paras)

    # example of how to perform matrix operations between NN at each lattice site simultaneously
    print(np.sum(all_neighbor_paras, axis=3)) 

# test_NN()


##### Check that leapfrog is reversible #####
def test_leapfrog():
    N = 16
    ell = 17 # increase ell at fixed eps to increase error
    eps = 0.7 # 1/ell
    model = SU2xSU2(N, a=1, ell=ell, eps=eps, beta=1)

    np.random.seed(6)
    # cold
    # a0 = np.ones((N,N,1))
    # ai = np.zeros((N,N,3))
    # phi_start = np.concatenate([a0,ai], axis=2)

    # hot
    a = np.random.standard_normal((N,N,4))
    phi_start = SU2.renorm(a)
    
    # normal
    # pi_start = np.random.standard_normal((N,N,3))
    # phi_end, pi_end = model.leapfrog(phi_start, pi_start)
    # phi_start_new, pi_start_new = model.leapfrog(phi_end, -pi_end)

    # FA
    model.A = model.kernel_inv_F()
    pi_start = model.pi_samples()
    phi_end, pi_end = model.leapfrog_FA(phi_start, pi_start)
    phi_start_new, pi_start_new = model.leapfrog_FA(phi_end, -pi_end)


    phi_delta = np.abs(phi_start_new-phi_start)
    pi_delta = np.abs(pi_start_new+pi_start)

    print('phi error:')
    print('Total: ', np.sum(phi_delta))
    print('Per site avg: ', 1/N**2 * np.sum(phi_delta))
    print('Biggest: ', np.max(phi_delta))

    print('\npi error:')
    print('Total: ', np.sum(pi_delta))
    print('Per site avg: ', 1/N**2 * np.sum(pi_delta))
    print('Biggest: ', np.max(pi_delta))

# test_leapfrog()


##### Check equipartion #####
def test_equi():
    '''SU(2) matrix has 3 DoF. Hence expect by equipartition (with k_b T = 1) that the average KE per site is 3*1/2.'''
    def KE_per_site(pi, N):
        K = 1/2 * np.sum(pi**2)
        return K/N**2
    
    N, ell, eps = 16, 10, 0.1 
    M = 2000
    model = SU2xSU2(N, a=1, ell=ell, eps=eps, beta=1)
    configs = np.empty((M+1, N, N, 4))
    momenta = np.empty((M, N, N, 3))
    kins = np.empty(M)

    # cold start
    a0 = np.ones((N,N,1))
    ai = np.zeros((N,N,3))
    configs[0] = np.concatenate([a0,ai], axis=2)
   
    for i in range(1,M+1):
        phi = configs[i-1]
        pi = np.random.standard_normal((N,N,3))
        phi_new, pi_new = model.leapfrog(phi, pi)

        delta_H = model.Ham(phi_new,-pi_new) - model.Ham(phi,pi)
        acc_prob = np.min([1, np.exp(-delta_H)])

        if acc_prob > np.random.random():
            configs[i] = phi_new
            momenta[i-1] = pi_new
        else:
            configs[i] = phi 
            momenta[i-1] = pi

        kins[i] = KE_per_site(momenta[i-1], N)

    # reject 10% burn in
    burn_in_idx = int(M*0.1) 
    KE_avg = np.mean(kins[burn_in_idx:])
    KE_err = np.std(kins[burn_in_idx:]) / np.sqrt(kins[burn_in_idx:].shape)

    print('avg KE per site = %.5f +/- %.5f'%(KE_avg, KE_err))

# test_equi()


##### Check equipartion for acceleration#####
def test_equi_FA():
    '''SU(2) matrix has 3 DoF. Hence expect by equipartition (with k_b T = 1) that the average KE per site is 3*1/2.'''
    def KE_per_site(pi, N, A):
        pi_F_mag = np.sum( np.abs(np.fft.fft2(pi, axes=(0,1)))**2, axis=-1 ) # (N,N) find magnitude of FT of each component of momentum in Fourier space. Then sum over all 3 components
        T = 1/(2*N**2) * np.sum(pi_F_mag*A) # sum over momentum Fourier lattice
        return T/N**2
    
    N, ell, eps = 32, 4, 1/4 # found manually/ by using calibration routine 
    M = 1000
    model = SU2xSU2(N, a=0.3, ell=ell, eps=eps, beta=0.75)
    configs = np.empty((M+1, N, N, 4))
    momenta = np.empty((M, N, N, 3))
    kins = np.empty(M)

    A = model.kernel_inv_F()
    model.A = A

    # cold start: seems to have problems evolving away from starting configuration (get 0% acceptance). Hence use hot start.
    # a0 = np.ones((N,N,1))
    # ai = np.zeros((N,N,3))
    # configs[0] = np.concatenate([a0,ai], axis=2)
    a = np.random.standard_normal((N,N,4))
    configs[0] = SU2.renorm(a)

    n_acc = 0
    with alive_bar(M) as bar:
        for i in range(1,M+1):
            phi = configs[i-1]
            pi = model.pi_samples()
            phi_new, pi_new = model.leapfrog_FA(phi, pi)

            delta_H = model.Ham_FA(phi_new,-pi_new) - model.Ham_FA(phi,pi)
            acc_prob = np.min([1, np.exp(-delta_H)])

            if acc_prob > np.random.random():
                n_acc += 1
                configs[i] = phi_new
                momenta[i-1] = pi_new
            else:
                configs[i] = phi 
                momenta[i-1] = pi

            kins[i-1] = KE_per_site(momenta[i-1], N, A)
            bar()

    print('acc rate = %.2f%%'%(n_acc/M*100))
    # reject 10% burn in
    burn_in_idx = int(M*0.1) 
    KE_avg = np.mean(kins[burn_in_idx:])
    KE_err = np.std(kins[burn_in_idx:]) / np.sqrt(kins[burn_in_idx:].shape)

    print('avg KE per site = %.5f +/- %.5f'%(KE_avg, KE_err))

# test_equi_FA()


##### Check disordered phase #####
def test_avg_components():
    '''In O(4) interpretation, each site hosts a 4D vector. The system is in the disordered phase for beta!=infty 
    such that the components of the vector when averaged over configurations and sites vanish'''

    model = SU2xSU2(N=16, a=1, ell=10, eps=0.1, beta=1)
    model.run_HMC(5000, 20, 0.1, store_data=False)    

    m_avg , m_err = model.order_parameter(make_plot=False)
    for i,(avg, err) in enumerate(zip(m_avg, m_err)):
            print('<m_%d> : %.5f +/- %.5f'%(i, avg, err))

# test_avg_components()


##### plot residual between simulation and coupling expansions #####
def residual_coupling():
    res = np.loadtxt('data/coupling_expansion')
    betas = res[:,0]
    e_avg = res[:,4]
    e_err = res[:,5]

    fig = plt.figure(figsize=(8,6))

    mask_s = betas<1
    b_s = betas[mask_s]
    strong = 1/2*b_s +1/6*b_s**3 +1/6*b_s**5

    mask_w = betas>0.6
    b_w = betas[mask_w]
    Q1 = 0.0958876
    Q2 = -0.0670
    weak = 1 - 3/(8*b_w) * (1 + 1/(16*b_w) + (1/64 + 3/16*Q1 + 1/8*Q2)/b_w**2)

    plt.errorbar(b_s, e_avg[mask_s]-strong, yerr=e_err[mask_s], color='g', fmt='.', capsize=2, label='HMC - s.c.')
    plt.errorbar(b_w, e_avg[mask_w]-weak, yerr=e_err[mask_w], color='b', fmt='.', capsize=2, label='HMC - w.c.')
    plt.legend(prop={'size': 12})
    plt.xlabel(r'$\beta$')
    plt.ylabel(r'residual')

    plt.show()

    return

# residual_coupling()


##### naive and FFT based computation of wall wall correlations #####
def ww_naive(model):
    '''
    Naive approach to compute the correlation function.
    '''
    def ww_correlation(i, j, m, model):
        '''correlates ith and jth column of lattice, defined as the average point to point correlation for all points contained in the walls'''
        pp_corrs = np.empty(model.N**2)
        for p in range(model.N):
            for q in range(model.N):
                # correlate matrices A and B at points (p,i) and (q,j). Reshape to use routines
                A = model.configs[m,p,i].reshape((1,1,4))
                B = model.configs[m,q,j].reshape((1,1,4))

                k = p*model.N + q
                prod = SU2.dot(A, SU2.hc(B))
                pp_corrs[k] = SU2.tr(prod + SU2.hc(prod))

        return np.mean(pp_corrs)


    L = model.N # largest considered separation 
    ww_cor = np.empty(L+1) # wall wall correlation for different separations
    ww_cor_err = np.empty(L+1)
    ds = np.arange(L+1)
    t1 = time.time()
    for d in ds:
        # smaller errors when using each wall wall pair as data point to estimate mean and error
        # all_ww_pairs = np.empty((model.M, model.N)) # each row contains all wall wall correlations at fixed d for a different configuration
        # for m in range(model.M):
        #     for i in range(model.N):
        #         all_ww_pairs[m,i] = ww_correlation(i, (i+d)%model.N, m, model)
            
        # ww_cor[d], _, ww_cor_err[d], _ = jackknife_stats(all_ww_pairs.flatten(), np.mean)
        avg_ww_configs = np.empty(model.M) # average wall wall correlation of each configuration
        for m in range(model.M):
            avg_ww_pairs = np.empty(model.N) # average wall wall correlation from all pairs in a single configuration
            for i in range(model.N):
                avg_ww_pairs[i] = ww_correlation(i, (i+d)%model.N, m, model)
            avg_ww_configs[m] = np.mean(avg_ww_pairs) 
        ww_cor[d], _, ww_cor_err[d], _ = jackknife_stats(avg_ww_configs, np.mean)
        print('d=%d done'%d)
    # normalize
    ww_cor, ww_cor_err = ww_cor/ww_cor[0], ww_cor_err/ww_cor[0]
    t2 = time.time()

    return ww_cor, ww_cor_err, t2-t1

def ww_fast(model):
    ''' 
    Computes the wall to wall correlation as described in the report via the cross correlation theorem.
    '''
    # for nicer plotting and fitting include value for separation d=self.N manually as being equivalent to d=0 (using periodic boundary conditions)
    ds = np.arange(model.N+1)
    ww_cor, ww_cor_err = np.zeros(model.N+1),np.zeros(model.N+1)
    ww_cor_err2 = np.zeros(model.N+1)

    t1 = time.time()
    Phi = np.sum(model.configs, axis=1) # (M,N,4)
    for k in range(4):
        # passes (N,M) arrays: correlations along axis 0 while axis 1 hosts results from repeating the measurement for different configurations
        cf, cf_err = correlations.correlator_repeats(Phi[:,:,k].T, Phi[:,:,k].T)
        ww_cor[:-1] += cf
        ww_cor_err2[:-1] += cf_err**2 # errors coming from each component of the parameter vector add in quadrature
    ww_cor *= 4/model.N**2
    ww_cor_err = 4/model.N**2 * np.sqrt(ww_cor_err2)
    ww_cor[-1], ww_cor_err[-1] = ww_cor[0], ww_cor_err[0]
    # normalize
    ww_cor, ww_cor_err = ww_cor/ww_cor[0], ww_cor_err/ww_cor[0]
    t2 = time.time()

    return ww_cor, ww_cor_err, t2-t1


def compare_ww():
    '''Compares correlation function and its error from the naive and cross correlation theorem based approach.
    '''
    paras = np.loadtxt('data/single_run/model_paras.txt')
    sim_paras = np.loadtxt('data/single_run/sim_paras.txt')
    print('Loading simulation:\nN, a, ell, eps, beta\n',paras,'\nM, thin freq, burn in, accept rate\n',sim_paras)
    model = SU2xSU2(*paras)
    model.load_data()

    ds = np.arange(model.N+1)
    cor_func_naive, cor_func_err_naive, _ = ww_naive(model)
    cor_func_fast, cor_func_err_fast, _ = ww_fast(model) 

    # difference in function values
    fig = plt.figure(figsize=(8,6))

    plt.plot(ds, 1 - cor_func_fast/cor_func_naive)
    plt.xlabel(r'lattice separation [$a$]')
    plt.ylabel(r'$1-C_{cross} / C_{sum}$')
    fig.gca().xaxis.set_major_locator(MaxNLocator(integer=True)) # set major ticks at integer positions only
    plt.show()

    # difference in errors
    fig = plt.figure(figsize=(8,6))

    plt.plot(ds, cor_func_err_naive, c='k', label='double sum')
    plt.plot(ds, cor_func_err_fast, c='g', label='cross correlation')
    plt.xlabel(r'lattice separation [$a$]')
    plt.ylabel('wall-wall correlation error')
    plt.legend(prop={'size':12})
    fig.gca().xaxis.set_major_locator(MaxNLocator(integer=True)) # set major ticks at integer positions only
    plt.show()

# compare_ww()
    

##### naive and FFT based computation of susceptibility #####
def susceptibility_naive(phi):
        '''
        Computes the susceptibility for lattice configuration phi.
        phi: (N,N,4) array
            parameter values of SU(2) matrices at each lattice site

        Returns
        chi: float
            the susceptibility
        '''
        N = phi.shape[0]
        # find product of phi with phi at every other lattice position y
        # phi_y is obtained by shifting the lattice by one position each loop
        G = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                phi_y = np.roll(phi, shift=(i,j), axis=(0,1))
                A = SU2.dot(phi, SU2.hc(phi_y))
                G += SU2.tr(A + SU2.hc(A))

        chi = np.sum(G) / (2*N**2)

        return chi

def susceptibility_fast(phi):
    '''
    Computes the susceptibility i.e. the average point to point correlation for configuration phi.
    As described in the report, this closely related to summing the wall to wall correlation function which can be computed efficiently via the cross correlation theorem.

    Returns:
    chi: float
        susceptibility of the passed configuration phi
    '''
    N = phi.shape[0]
    ww_cor = np.zeros(N)
    Phi = np.sum(phi, axis=0) # (N,4)
    for k in range(4):
        cf, _ = correlations.correlator(Phi[:,k], Phi[:,k])
        ww_cor += cf
    ww_cor *= 2/N**2
    chi = np.sum(ww_cor)

    return chi


def compute_chi():
    '''Compares naive double sum and cross correlation theorem approach to computing the susceptibility.
    '''
    N = 64

    # test extreme case in which chi=2*N^2
    # config = np.zeros((model.N,model.N,4))
    # config[:,:,0] = 1
    # random lattice configuration
    a = np.random.standard_normal((N,N,4))
    config = SU2.renorm(a)
    # choose a configuration manually from the chain
    # model = SU2xSU2(N=N, a=1, ell=10, eps=1/10, beta=1)
    # model.run_HMC(20, 1, 0, accel=False, store_data=False) 
    # config = model.configs[-10]
    
    t1 = time.time()
    chi_cross_cor  = susceptibility_fast(config)
    t2 = time.time()
    print('cross_cor result: ',chi_cross_cor)
    print('cross_cor time: %s'%(str(timedelta(seconds=t2-t1))))

    t1 = time.time()
    chi_naive  = susceptibility_naive(config)
    t2 = time.time()
    print('naive result: ',chi_naive)
    print('naive time: %s'%(str(timedelta(seconds=t2-t1))))

# compute_chi()


def chi_speed_compare():
    '''
    Makes plot to compare speed of chi computation using naive double sum or the cross correlation theorem.
    '''
    Ns = np.linspace(10, 512, num=15, endpoint=True, dtype=int) # naive method can take upto 1e4 sec for single calculation at N approx 400 
    ts_crosscor = np.empty_like(Ns, dtype=float)
    ts_naive = np.empty_like(Ns, dtype=float)

    for i,N in enumerate(Ns):
        a = np.random.standard_normal((N,N,4))
        phi = SU2.renorm(a)
    
        t1 = time.time_ns() # nessessary to capture run time at small N
        chi_cross_cor  = susceptibility_fast(phi)
        t2 = time.time_ns()
        ts_crosscor[i] = t2-t1

        t1 = time.time()
        chi_cross_cor  = susceptibility_naive(phi)
        t2 = time.time()
        ts_naive[i] = t2-t1
        print('Completed N = ',N)

    fig = plt.figure(figsize=(8,6))
    plt.plot(Ns, ts_naive, c='k', label='double sum')
    plt.plot(Ns, ts_crosscor, c='g', label='cross correlation')
    plt.xlabel('lattice size N')
    plt.ylabel('CPU time [sec]')
    plt.yscale('log')
    plt.legend(prop={'size': 12})
    plt.show()
    # fig.savefig('plots/chi_speed.pdf')

    data = np.row_stack((Ns, ts_crosscor, ts_naive))
    np.savetxt('data/chi_speed.txt', data, header='lattice size N, CPU time via cross correlation thm, CPU time via double sum')

# chi_speed_compare()