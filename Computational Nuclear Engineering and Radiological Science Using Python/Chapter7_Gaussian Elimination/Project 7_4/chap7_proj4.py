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

def inverseMat_gauss(A):
    '''
    Returns the inverse of the input matrix. Uses gaussian elimination

    Parameters
    ----------
    A : numpy matrix
        input matrix.

    Returns
    -------
    sol : numpy matrix
          inverse of the input matrix

    '''
    n,_ = A.shape
    
    const_vec = np.identity(n)
    sol = np.zeros(shape = (n,n))
    
    for i in range(n):
        sol[:, i] = gauss_elim_gen(A, const_vec[:, i])
    
    return sol
        

A = np.random.rand(5,5)
inv_A = inverseMat_gauss(A)
print(A@inv_A) #check to see if this returns the identity matrix

