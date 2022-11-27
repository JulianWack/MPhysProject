import numpy as np
import timeit
import SU2_mat_routines as SU2


##### Check if my matrix routine gives same result as np #####
def test_SU2_mat():
    # np.random.seed(42)
    aa = 10*np.random.random(4)
    bb = -66.8*np.random.random(4)

    my_A = aa
    my_B = bb

    def make_np_mat(a):
        return np.matrix( [[a[0]+1j*a[3], a[2]+1j*a[1]], [-a[2]+1j*a[1], a[0]-1j*a[3]]] )

    np_A = make_np_mat(aa)
    np_B = make_np_mat(bb)

    my_C = SU2.make_mat( SU2.dot(my_A, my_B) )
    np_C = np.matmul(np_A, np_B)

    # need to convert np.matrix object into ndarray (via .A) for allclose comparison 
    print('Same matrix product: ', np.allclose(my_C.A, np_C.A))
    my_hc = SU2.make_mat(SU2.hc(my_A))
    print('Same hc: ', np.allclose(my_hc.A, np_A.H.A))
    print('Same det: ', np.allclose(SU2.det(my_A), np.linalg.det(np_A.A).real))
    print('Same trace: ', np.allclose(SU2.tr(my_A), np.trace(np_A.A).real))


    ##### Compare speed of matrix multiplication #####
    # tabbing important to assure that string has no tabs. Otherwise timeit throws an error
    # when using timeit outside a function, can tab all lines to the same height

    set_up = '''
import numpy as np
import SU2_mat_routines as SU2
'''

    my_test_code = ''' 
aa = np.random.random(4)
bb = np.random.random(4)

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

    n_iter = 100000
    # print total time needed to perform n_iter executions of the test code
    print('My product: ', timeit.timeit(setup=set_up, stmt=my_test_code, number=n_iter))
    print('np product: ', timeit.timeit(setup=set_up, stmt=np_test_code, number=n_iter))

test_SU2_mat()
########
