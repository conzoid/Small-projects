'''
simple_LU_decomposition(A) returns L and U only without pivoting
if the solution is wanted, use LU_solve(L, U, b)
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


def simple_LU_decomposition(A):
  '''
  simple LU decomposition function without any pivoting
  
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
  
  #reduction
  for i in range(N):
    for j in range(i+1 , N):
      M[j, i, i] = -1 * (A[j,i] / A[i,i])
      A[j] -= (A[j,i] / A[i,i]) * A[i] 

  
  L = np.identity(N) #this is to store the inv_L mat, later we will inverse it
  for i in range(N-2, -1, -1):
    L = np.dot(L, M[:,:,i])
  L = np.linalg.inv(L)  
    
  return A, L

def LU_solve(L, U, b):
  '''
  returns the solution vector 'x' of LUx = b
  
  :param L: lower triangle matrix
  :type L: numpy matrix
  
  :param U: upper triangle matrix
  :type U: numpy matrix
  
  :param b: constant vector
  :type b: numpy array
  
  :return: solution vector
  :rtype: numpy array

  '''
  
  y = front_substitution(L, b)
  x = back_substitution(U, y)
  
  return x

#example solution
A = np.matrix([
  (3,2,1),
  (-1,4,5),
  (3,5,3)
  ])
b = np.array([1,6,4])

[U,L] = simple_LU_decomposition(A)
x = LU_solve(L, U, b)

