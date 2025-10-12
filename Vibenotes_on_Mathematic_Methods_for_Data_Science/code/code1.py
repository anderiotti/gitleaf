import numpy as np

def jacobi_solve(A, b, x0, max_iter, tol=1e-6):
    """
    Solves Ax = b using the Jacobi method. M = D, N = L + U.
    Implementation is highly vectorized (O(n^2) or O(n) for sparse A).
    """
    n = A.shape[0]
    x_old = np.array(x0, dtype=A.dtype) # Initialize iterate
    
    # Extract D and R = L + U (R is off-diagonal)
    D = np.diag(A)
    R = A - np.diag(D)
    
    for k in range(1, max_iter + 1):
        # x_new = D^{-1} * (b - R * x_old)
        # @ is matrix multiplication; / D is component-wise division (D^{-1} operation)
        x_new = (b - R @ x_old) / D 
        
        # Relative Error Criterion (L_inf norm)
        error_norm = np.linalg.norm(x_new - x_old, np.inf)
        if error_norm < tol * np.linalg.norm(x_new, np.inf):
            # print(f"Jacobi converged in {k} iterations.")
            return x_new, k
            
        x_old = x_new
        
    # print(f"Jacobi failed to converge in {max_iter} iterations.")
    return x_old, max_iter