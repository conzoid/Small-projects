import numpy as np

def complete_pivot(mat, order, num):
  '''
  complete pivoting in the submatrix of mat indicated by num
  
  :param mat: augmented matrix to be pivoted
  :type mat: numpy matrix
  
  :param order: keeps track of column swaps
  :type order: numpy array
  
  :param num: index of diagonal
  :type num: integer
  
  :return: changes the matrices order and mat
  :rtype: nothing

  '''
  N,_ = mat.shape
  
  if np.max(mat[num:, num:N]) > np.absolute(np.min(mat[num:, num:N])):   #checks the absolute max 
    row, col = np.unravel_index(mat[num:, num:N].argmax(), mat[num:, num:N].shape)
    row += num
    col += num
    
    mat[:, [num, col]] = mat[:, [col, num]]
    mat[[num, row]] = mat[[row, num]]
    order[[num, col]] =  order[[col, num]]
  else:
    row, col = np.unravel_index(mat[num:, num:N].argmin(), mat[num:, num:N].shape)
    row += num
    col += num
    
    mat[:, [num, col]] = mat[:, [col, num]]
    mat[[num, row]] = mat[[row, num]]
    order[[num, col]] =  order[[col, num]]
     
  return

def gauss_elim_gen(A, b):
  '''
  General gaussian N*N solver involving complete pivoting
  
  Parameters
  ----------
  A : numpy N*N array
      this is the coefficient matrix.
  b : numpy array
      constant vector.
  
  Returns
  -------
  y : numpy array
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
  order = np.arange(N) #keeps track of column swaps
  y = np.zeros(N)
  

  #reduction
  for i in range(N):
    complete_pivot(mat, order, i)
    for j in range(i+1 , N):
      mat[j] -= (mat[j,i] / mat[i,i]) * mat[i] 
  
  #back substitution
  for i in list(range(N-1, -1, -1)):
    x[i] = mat[i,N]
    
    for j in range(i+1, N):
      x[i] -= mat[i, j] * x[j]
    x[i] /= mat[i,i]
  
  y[order] = x      #fix the order

  return y



# simple test for 100 random 6by6 cases
items = 100
for i in range(items):
    A = np.random.rand(5,5)
    x = np.random.rand(5)
    b = np.dot(A, x)
    sol = gauss_elim_gen(A, b)
    
    try:
        assert np.max(np.abs(sol - x)) < 1e-8 #set a tolerance for error
    except AssertionError:
        print("Test failed with")
        print("A = ", A)
        print("x = ", x)
        print(np.max(np.abs(sol - x)))
        raise
print("All tests passed")


