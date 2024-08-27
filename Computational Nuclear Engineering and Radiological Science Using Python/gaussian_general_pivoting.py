import numpy as np
def pivot(mat):
    '''
    Function used for pivoting the augmented matrix

    Parameters
    ----------
    mat : numpy matrix
        augmented matrix

    Returns
    -------
    None.
    
    By-Product
    ----------
    Changes the input matrix mat
    '''
    
    N,_ = mat.shape
    
    for i in range(N):
        if np.max(mat[i:, i]) > np.absolute(np.min(mat[i:, i])):   #checks the absolute max 
            max_index = i + mat[i:, i].argmax()
        else:
            max_index = i + mat[i:, i].argmin()
        mat[[i, max_index]] = mat[[max_index, i]] 
    
    return
    
def gauss_elim_gen(A, b):
    '''
    General simple gaussian N*N solver involving pivoting
    
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
    
    #setting up augmented matrix
    mat = np.zeros([N,N+1])
    mat[:, 0:N] = A
    mat[:, N] = b

    pivot(mat)
    
    #reduction
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


#simple test for 100 random 6by6 cases
items = 100
for i in range(items):
    A = np.random.rand(6,6)
    x = np.random.rand(6)
    b = np.dot(A, x)
    sol = simple_gauss_elim_gen(A, b)
    
    try:
        assert np.max(np.abs(sol - x)) < 1e-12
    except AssertionError:
        print("Test failed with")
        print("A = ", A)
        print("x = ", x)
        raise
print("All tests passed")


#this code works most of the time, however there were some exceptions
 
