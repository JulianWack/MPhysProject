# Several functions to perform common matrix operations for SU(2) matrices.
# By exploiting their properties, these routines are more efficient than general matrix methods
# Specifically, an SU(2) matrix is fully specified by 4 real parameters:
# [[a0 + i*a3, a2 + i*a1],
# [-a2 + i*a1, a0 - i*a3]]


import numpy as np


def alpha_to_a(alpha):
    '''Convert parameters of SU(2) matrix when given in terms of the exponential map to the parameters when given as linear combination of gens and unity:
    U = exp(i alpha_i sigma_i) = a_0*1 + i*a_i sigma_i
        
    alpha: 1D array length 3
        parameters when representing a SU(2) group element via the exponential map

    Returns:
    a: 1D array length 4
        parameters when explicitly evaluating the exponential map
    '''
    a = np.empty(4)
    alpha_norm = np.sqrt(np.sum(alpha))
    alpha_unitvec = alpha / alpha_norm
    a[0] = np.cos(alpha_norm)
    a[1:] = alpha_unitvec * np.sin(alpha_norm)

    return a


def make_mat(a):
    '''Constructs full matrix explicitly.
    '''
    mat = [[a[0]+1j*a[3], a[2]+1j*a[1]], 
    [-a[2]+1j*a[1], a[0]-1j*a[3]]]

    return np.matrix(mat)


### basic quantities ###
def hc(a):
    '''Returns parameter vector of the hermitian conjugate.
    '''
    new_a = -a 
    new_a[0] = a[0]

    return new_a


def tr(a):
    '''Returns trace
    '''
    return 2*a[0]


def det(a):
    '''determinant is given by the squared length of the parameter vector
    '''
    return norm2(a)


def norm2(a):
    '''Returns squared norm of parameter vector
    '''
    return np.sum(a**2)


def renorm(a):
    '''Renormalises matrix to have det = 1
    '''
    return a / np.sqrt(norm2(a))


### combining two SU(2) matrices ### 
def dot(a, b):
    '''Computes matrix product A.B when matrices A and B have associated parameter vectors a and b.
    '''
    c0 = a[0]*b[0] - np.sum(a[1:]*b[1:])
    c1 = a[0]*b[1] + a[1]*b[0] + a[3]*b[2] - a[2]*b[3]
    c2 = a[0]*b[2] + a[2]*b[0] + a[1]*b[3] - a[3]*b[1]
    c3 = a[0]*b[3] + a[3]*b[0] + a[2]*b[1] - a[1]*b[2]
    c = np.array([c0, c1, c2, c3])

    return c


def sum(a, b):
    '''Computes sum of two SU(2) matrices A and B with associated parameter vectors a and b.
    Let C = A + B, i.e. c = a + b. 
    Note that the sum of two SU(2) matrices is proportional to an SU(2) matrix with proportionality constant k, meaning
    D = C/k = 1/k (A + B) is in SU(2).
    To only having to perform manipulations on SU(2) matrices, the parameters d of the SU(2) matrix D and the constant k is returned such that their product gives the 
    parameter vector of C, the sum of A and B.
    '''
    c = a + b
    k_2 = 2*(a[0]*b[0] + a[1:]*b[1:] + 1)

    k = np.sqrt(k_2)
    d = c / k

    return d, k