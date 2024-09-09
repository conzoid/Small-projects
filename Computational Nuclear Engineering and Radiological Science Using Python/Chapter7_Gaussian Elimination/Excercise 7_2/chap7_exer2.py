import numpy as np

def partial_pivot(mat, index):
    '''
    Function used for partial pivoting the augmented matrix

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
    

    if np.max(mat[index:, index]) > np.absolute(np.min(mat[index:, index])):   #checks the absolute max 
        max_index = index + mat[index:, index].argmax()
    else:
        max_index = index + mat[index:, index].argmin()
    mat[[index, max_index]] = mat[[max_index, index]] 

    return

def gauss_elim_gen(A, b):
    '''
    General simple gaussian N*N solver involving partial pivoting
    
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
    
    #reduction
    for i in range(N):
        partial_pivot(mat, i)
        for j in range(i+1 , N):
            mat[j] -= (mat[j,i] / mat[i,i]) * mat[i] 
    
    #back substitution
    for i in list(range(N-1, -1, -1)):
        x[i] = mat[i,N]
        
        for j in range(i+1, N):
            x[i] -= mat[i, j] * x[j]
        x[i] /= mat[i,i]
    
    return x

def set_up_mat(n):
    '''
    Creates the required matrices for exerise 7.2

    Parameters
    ----------
    n : integer
        size of required matrices.

    Returns
    -------
    A : np matrix
        coefficient matrix.
    b : np matrix
        constant vector.

    '''
    assert n>0
    
    # #initializing required matrices
    A = np.zeros((n,n))
    b = np.zeros(n)
    dummy_mat = np.arange(0, n)
    
    #setting up 'b'
    even_values = dummy_mat[dummy_mat%2 == 0]
    odd_values = dummy_mat[dummy_mat%2 == 1]
    b[even_values] = 1
    b[odd_values] = -1
    
    #setting up 'A'
    for i in range(n):
        A[i] = np.add(np.add(dummy_mat, 1), i)
    A = np.reciprocal(A, dtype = float)
    
    return A, b

#test
tests = 10000
for i in range(1, tests):
    A, b = set_up_mat(i)
    x = gauss_elim_gen(A, b) 
    diff = np.max(np.abs(b - np.dot(A, x)))
    max_index = np.argmax(np.abs(b - np.dot(A, x)))
    perc_error = (diff/x[max_index]) * 100
    try:
        assert perc_error < 1e-2
    except AssertionError:
        print("Error for i = ", i, "with perc = ", perc_error)
        raise
print("All tests passed!")

