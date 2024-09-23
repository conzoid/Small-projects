'''
PLU_decomposition(A) returns P, L and U only with partial pivoting
PLU_solve(P, L, U, b) returns the solution of PLUx = b where PLU = A
'''

import numpy as np

def front_substitution(A, b):
  '''
  Solves the equation Ax = b where A is a lower triangle matrix
  
  :param A: lower triangle matrix
  :type A: numpy matrjix
  
  :param b: constant vector
  :type b: numpy array
  
  :return: solution vector
  :rtype: numpy array

  '''
  
  _,N = A.shape
  x = np.zeros(N)   #solution vector
  mat = np.zeros([N, N+1])    #augmented matrix
  mat[:, 0:N] = A
  mat[:, N] = b
  
  for i in range(N):
    x[i] = mat[i,N]
    
    for j in range(i):
        x[i] -= mat[i, j] * x[j]
    x[i] /= mat[i,i]

  return x


def back_substitution(A, b):
  '''
  Solves the equation Ax = b where A is an upper triangle matrix
  
  :param A: upper triangle matrix
  :type A: numpy matrjix
  
  :param b: constant vector
  :type b: numpy array
  
  :return: solution vector
  :rtype: numpy array

  '''
  
  _,N = A.shape
  x = np.zeros(N)   #solution vector
  mat = np.zeros([N, N+1])    #augmented matrix
  mat[:, 0:N] = A
  mat[:, N] = b
  
  for i in range(N-1, -1, -1):
    x[i] = mat[i,N]
    
    for j in range(i+1, N):
        x[i] -= mat[i, j] * x[j]
    x[i] /= mat[i,i]

  return x

def partial_pivot(mat, index):
  '''
  Function used for partial pivoting the augmented matrix

  Parameters
  ----------
  mat : numpy matrix

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

  return max_index


def PLU_decomposition(A):
  '''
  LU decomposition function with partial pivoting
  factorizes  the matrix A into PLU where P is a permutaiton matrix
  
  :param A: N*N coefficient matrix
  :type A: numpy matrix
  
  :return: L, U matrices
  :rtype: numpy matrices

  '''
  A = A.astype(float)
  [A_row, A_column] = A.shape
  assert A_row == A_column
  
  N = A_row
  M = np.dstack([np.identity(N)] * (N-1))  #elementary matrices
  p = np.dstack([np.identity(N)] * (N-1))  #permutation matrices
  U = A.copy()
  
  #reduction
  for i in range(N-1):
    max_ind = partial_pivot(U, i)
    p[i, i, i] = 0
    p[max_ind, max_ind, i] = 0
    
    p[i, max_ind, i] = 1
    p[max_ind, i, i] = 1
    
    for j in range(i+1 , N):
      M[j, i, i] = -1 * (U[j,i] / U[i,i])
      U[j] -= (U[j,i] / U[i,i]) * U[i] 
  
  L = np.identity(N) #this is to store the inv_L mat, later we will inverse it
  for i in range(N-2, 0, -1):
    L = np.dot(L, M[:,:,i])
    L = np.dot(L, p[:,:,i])
  L = np.dot(L, M[:,:,0])
  for i in range(1, N-1):
    L = np.dot(L, p[:,:,i])
  L = np.linalg.inv(L)  
  
  P = np.identity(N)  #final permutaiton matrix
  for i in range(N-1):
    P = np.dot(P, p[:,:,i])
    
  return U, L, P

def PLU_solve(P, L, U, b):
  '''
  returns the solution vector 'x' of PLUx = b
  
  :param P: permutation matrix
  :type P: numpy matrix
  
  :param L: lower triangle matrix
  :type L: numpy matrix
  
  :param U: upper triangle matrix
  :type U: numpy matrix
  
  :param b: constant vector
  :type b: numpy array
  
  :return: solution vector
  :rtype: numpy array

  '''
  
  y = front_substitution(L, np.dot(np.transpose(P), b))
  x = back_substitution(U, y)
  
  return x


#example solution
A = np.random.rand(6,6)
b = np.random.rand(6)
[U,L,P] = PLU_decomposition(A)
x = PLU_solve(P, L, U, b)



