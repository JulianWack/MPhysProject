import numpy as np
import timeit
import SU2_mat_routines as SU2


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
