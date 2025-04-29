import numpy as np
import random

random.seed(22)

def dot_product(u:np.ndarray, v:np.ndarray) -> float:
    """ Compute the dot product of two vectors u and v. """
    res = 0.0
    for i in range(u.shape[0]):
        res += u[i] * v[i]
    return res

def norm(u:np.ndarray) -> float:
    """ Compute the Euclidean norm of vector u. """
    return (dot_product(u, u))** 0.5

def matrix_vector_multiplication(A:np.ndarray, x:np.ndarray) -> np.ndarray:
    """ Compute the matrix-vector product Ax. """
    m, n = A.shape
    y = np.zeros(m, dtype=float)
    for i in range(m):
        res = 0.0
        for j in range(n):
            res += A[i, j] * x[j]
        y[i] = res
    return y

def transpose(A:np.ndarray) -> np.ndarray:
    """ Compute the transpose of matrix A. """
    m, n = A.shape
    T = np.zeros((n, m), dtype=float)
    for i in range(m):
        for j in range(n):
            T[j, i] = A[i, j]
    return T

def matrix_multiplication(A:np.ndarray, B:np.ndarray) -> np.ndarray:
    """ Compute the matrix product AB. """
    m, p = A.shape
    p2, n = B.shape
    assert p == p2
    C = np.zeros((m, n), dtype=float)
    for i in range(m):
        for j in range(n):
            s = 0.0
            for k in range(p):
                s += A[i, k] * B[k, j]
            C[i, j] = s
    return C

def outer(u:np.ndarray, v:np.ndarray, scale:float=1.0) -> np.ndarray:
    """ Compute the outer product of vectors u and v, scaled by scale. """
    m = u.shape[0]
    n = v.shape[0]
    M = np.zeros((m, n), dtype=float)
    for i in range(m):
        for j in range(n):
            M[i, j] = scale * u[i] * v[j]
    return M

def power_iteration(B:np.ndarray, num_iters:int=1000, tol:int=1e-10):
    """ Compute the dominant eigenvalue and eigenvector of matrix B using power iteration. """
    n, n2 = B.shape
    assert n == n2

    v = np.array([random.random() for _ in range(n)], dtype=float)
    v /= norm(v)

    lambda_old = 0.0
    for _ in range(num_iters):
        w = matrix_vector_multiplication(B, v)
        w_norm = norm(w)
        v = w / w_norm
        lambda_new = dot_product(v, matrix_vector_multiplication(B, v))
        if abs(lambda_new - lambda_old) < tol:
            break
        lambda_old = lambda_new

    return lambda_new, v

def svd_full(A:np.ndarray, num_iters:int=1000, tol:int=1e-10):
    """ Compute the Singular Value Decomposition (SVD) of matrix A using power iteration. """
    m, n = A.shape
    A_copy = A.copy().astype(float)
    U_part, S_vals, V_part = [], [], []

    r = min(m, n)
    for _ in range(r):
        B = matrix_multiplication(transpose(A_copy), A_copy)
        lambda1, v1 = power_iteration(B, num_iters=num_iters, tol=tol)
        sigma = (max(lambda1, 0.0))** 0.5
        if sigma < tol:
            break
        Av = matrix_vector_multiplication(A_copy, v1)
        u1 = Av / sigma

        U_part.append(u1)
        S_vals.append(sigma)
        V_part.append(v1)

        A_copy = A_copy - outer(u1, v1, scale=sigma)

    rank = len(S_vals)


    U = np.column_stack(U_part)

    V = np.column_stack(V_part)

    Sigma = np.diag(S_vals)

    Vt = transpose(V)

    return U, Sigma, Vt