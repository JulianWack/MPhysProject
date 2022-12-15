import numpy as np
import timeit
import SU2_mat_routines as SU2
from SU2xSU2 import SU2xSU2


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
    ell = 10 # increase ell at fixed eps to increase error
    eps = 0.1 # 1/ell
    model = SU2xSU2(N, a=1, ell=ell, eps=eps, beta=1)

    # np.random.seed(6)
    # a0 = np.ones((N,N,1))
    # ai = np.zeros((N,N,3))
    ai = np.random.uniform(-1, 1, size=(N,N,3))
    a0 = (1 - np.sum(ai**2, axis=2)).reshape((N,N,1))
    phi_start = np.concatenate([a0,ai], axis=2)
    pi_start = np.random.standard_normal((N,N,3))

    phi_end, pi_end = model.leapfrog(phi_start, pi_start)
    phi_start_new, pi_start_new = model.leapfrog(phi_end, -pi_end)

    phi_delta = np.abs(phi_start_new-phi_start)
    pi_delta = np.abs(pi_start_new+pi_start)

    print('phi error:')
    print('Total: ', np.sum(phi_delta))
    print('Biggest: ', np.max(phi_delta))

    print('\npi error:')
    print('Total: ', np.sum(pi_delta))
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
   
    for i in range(1,M):
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