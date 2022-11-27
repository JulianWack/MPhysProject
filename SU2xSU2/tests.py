import numpy as np
import timeit
from SU2_matrix import *


##### Check if my matrix routine gives same result as np #####
def test_SU2_mat():
    # np.random.seed(42)
    aa = 10*np.random.random(4)
    bb = -66.8*np.random.random(4)

    my_A = SU2_mat(aa)
    my_B = SU2_mat(bb)

    def make_np_mat(a):
        return np.matrix( [[a[0]+1j*a[3], a[2]+1j*a[1]], [-a[2]+1j*a[1], a[0]-1j*a[3]]] )

    np_A = make_np_mat(aa)
    np_B = make_np_mat(bb)

    my_C = SU2_dot(my_A, my_B).make_mat()
    np_C = np.matmul(np_A, np_B)

    # need to convert np.matrix object into ndarray (via .A) for allclose comparison 
    print('Same matrix product: ', np.allclose(my_C.A, np_C.A))
    my_hc = my_A.hc().make_mat()
    print('Same hc: ', np.allclose(my_hc.A, np_A.H.A))
    print('Same det: ', np.allclose(my_A.det(), np.linalg.det(np_A.A).real))
    print('Same trace: ', np.allclose(my_A.tr(), np.trace(np_A.A).real))


    ##### Compare speed of matrix multiplication #####
    # tabbing important to assure that string has no tabs. Otherwise timeit throws an error
    # when using timeit outside a function, can tab all lines to the same height

    set_up = '''
import numpy as np
from SU2_matrix import SU2_mat, SU2_dot
'''

    my_test_code = ''' 
aa = np.random.random(4)
bb = np.random.random(4)

my_A = SU2_mat(aa)
my_B = SU2_mat(bb)

SU2_dot(my_A, my_B)
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

    n_iter = 100000
    # print total time needed to perform n_iter executions of the test code
    print('My product: ', timeit.timeit(setup=set_up, stmt=my_test_code, number=n_iter))
    print('np product: ', timeit.timeit(setup=set_up, stmt=np_test_code, number=n_iter))

test_SU2_mat()
########
