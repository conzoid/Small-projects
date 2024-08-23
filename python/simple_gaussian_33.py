import numpy as np

def simple_gauss_elim_33(A, b):
    '''
    Very simple gaussian 3*3 solver not involving any pivoting
    
    Parameters
    ----------
    A : numpy 3*3 array
        this is the coefficient matrix.
    b : numpy array
        constant vector.

    Returns
    -------
    x : numpy array
        solution vector.

    '''
    x = np.zeros([1,3]) #solution vector
    mat = np.zeros([3,4])
    mat[:, 0:3] = A
    mat[:, 3] = b

    for i in range(1, 3):
        for j in range(0,i):
            mat[i] -= (mat[i,j] / mat[j,j]) * mat[j] #making the lower left traingle 0s
    
    #back substitution
    for i in [2,1,0]:
        x[0,i] = mat[i,3]
        
        for j in range(i+1, 3):
            x[0,i] -= mat[i, j]
        x[0,i] /= mat[i,i]
    
    return x

A = np.matrix([(3.0,2,1),(-1,4,5),(2,-8,10)])
b = np.array([6,8,4])

print(simple_gauss_elim_33(A, b))

 
