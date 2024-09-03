import numpy as np

def simple_gauss_elim_gen(A, b):
    '''
    Very simple gaussian N*N solver not involving any pivoting
    
    Parameters
    ----------
    A : numpy N*N array
        this is the coefficient matrix.
    b : numpy array
        constant vector.

    Returns
    -------
    x : numpy array
        solution vector.

    '''
    [A_row, A_column] = A.shape
    assert A_row == A_column
    assert A_row == b.size
    
    N = A_row
    x = np.zeros(N) #solution vector
    mat = np.zeros([N,N+1])
    mat[:, 0:N] = A
    mat[:, N] = b

    for i in range(1, N):
        for j in range(0,i):
            mat[i] -= (mat[i,j] / mat[j,j]) * mat[j] #making the lower left traingle 0s
    
    #back substitution
    for i in list(range(N-1, -1, -1)):
        x[i] = mat[i,N]
        
        for j in range(i+1, N):
            x[i] -= mat[i, j] * x[j]
        x[i] /= mat[i,i]
    
    return x


y = np.arange(4)+1.0
A = np.array([(3.0,2,1,1),(-1,4,5,-2),(2,-8,10,-3),(2,3,4,5)])
b = np.dot(A,y)
    
sol = simple_gauss_elim_gen(A, b)


print(sol)

